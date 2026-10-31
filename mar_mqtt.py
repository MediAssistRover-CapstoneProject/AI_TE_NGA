"""MediAssistRover demo vision MQTT adapter.

MQTT v3.1.1 over TLS, publishes sanitized anomaly events and 1 Hz non-PII status.
This is a *vision adapter* topic, not an assertion of BE_MAR's robot event schema.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_event(event: dict, robot_id: str) -> dict:
    """Create a minimal, privacy-conscious MQTT payload; no frames/landmarks."""
    event_id = event.get("event_id") or event.get("id")
    event_type = event.get("event_type") or event.get("type")
    if not event_id or not event_type:
        raise ValueError("Event must have an ID and a type")

    result = {
        "event_id": str(event_id),
        "robot_id": str(robot_id),
        "event_type": str(event_type),
        "source": "VISION_AI",
        "detected_at": (
            event.get("detected_at")
            or event.get("time")
            or datetime.now(timezone.utc).isoformat()
        ),
        "algorithm": event.get("algorithm", "unspecified"),
        "confidence": event.get("confidence"),
        "simulation": True,  # Always sandbox until BE schema and model policy approved.
        "requires_staff_verification": True,
    }
    if event.get("person_id") is not None:
        result["person_id"] = int(event["person_id"])
    if isinstance(event.get("metrics"), dict):
        allowed_keys = ("angle_deg", "hip_speed", "angular_speed_deg", "aspect_ratio", "horizontal_duration_sec")
        clean_metrics = {
            k: round(float(event["metrics"][k]), 4)
            for k in allowed_keys
            if k in event["metrics"] and event["metrics"][k] is not None
        }
        if clean_metrics:
            result["metrics"] = clean_metrics
    if event.get("event_text_vi"):
        result["event_text_vi"] = str(event["event_text_vi"])
    if event.get("gesture_code"):
        result["gesture_code"] = str(event["gesture_code"])
    return result


class VisionMqttPublisher:
    """Async MQTT publisher: camera thread never waits for broker ACK."""

    def __init__(self):
        self.enabled = _bool("MQTT_ENABLED")
        self.robot_id = os.getenv("ROBOT_ID", "").strip()
        self.topic = os.getenv("MQTT_TOPIC", f"mar/robots/{self.robot_id}/vision/detections")
        self.status_topic = os.getenv(
            "MQTT_STATUS_TOPIC", f"mar/robots/{self.robot_id}/vision/status"
        )
        self.connected = threading.Event()
        self._stop = threading.Event()
        self._pending = queue.Queue(maxsize=128)
        self._seen = set()
        self._last_status = 0.0
        self._client = None
        self._worker = None
        self.last_error = None
        self.sent_events = 0
        self.ack_events = 0

        if not self.enabled:
            print("[MAR MQTT] disabled; local browser test only")
            return

        host = os.getenv("MQTT_HOST", "").strip()
        port = int(os.getenv("MQTT_PORT", "8883"))
        username = os.getenv("MQTT_USERNAME", "").strip()
        password = os.getenv("MQTT_PASSWORD", "")
        client_id = os.getenv("MQTT_CLIENT_ID", "mar-ai-vision")
        if not all((self.robot_id, host, username, password)):
            self.last_error = "Missing MQTT_HOST/MQTT_USERNAME/MQTT_PASSWORD/ROBOT_ID"
            self.enabled = False
            print("[MAR MQTT] CONFIG ERROR:", self.last_error)
            return

        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
            clean_session=True,
            transport="tcp",
        )
        client.username_pw_set(username=username, password=password)
        # TLS certificate and hostname validation remain enabled.
        ca = os.getenv("MQTT_CA_CERT", "").strip()
        client.tls_set(ca_certs=ca if ca else None)
        client.tls_insecure_set(False)
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        self._client = client

        try:
            client.connect_async(host, port, keepalive=60)
            client.loop_start()
            self._worker = threading.Thread(
                target=self._publish_worker,
                name="MAR-Vision-MQTT",
                daemon=True,
            )
            self._worker.start()
            print(f"[MAR MQTT] connecting TLS to {host}:{port}; topic={self.topic}")
        except Exception as exc:
            self.last_error = str(exc)
            print("[MAR MQTT] START ERROR:", type(exc).__name__, str(exc))

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if not reason_code.is_failure:
            self.connected.set()
            self.last_error = None
            print("[MAR MQTT] CONNECTED (TLS)")
        else:
            self.connected.clear()
            self.last_error = f"Broker rejected MQTT CONNECT: {reason_code}"
            print("[MAR MQTT]", self.last_error)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()
        if not self._stop.is_set():
            self.last_error = f"Disconnected: {reason_code}"
            print("[MAR MQTT]", self.last_error)

    def emit_event(self, event):
        """Enqueue once per event ID; QoS 1 delivery may still duplicate after retry."""
        if not self.enabled or not event:
            return
        try:
            payload = normalize_event(event, self.robot_id)
        except (ValueError, TypeError) as exc:
            print("[MAR MQTT] BAD EVENT:", exc)
            return
        event_id = payload["event_id"]
        if event_id in self._seen:
            return
        # No event-ID eviction needed for typical demo sessions. Restart to reset.
        try:
            self._pending.put_nowait(payload)
            self._seen.add(event_id)
        except queue.Full:
            self.last_error = "Event queue full; publish not guaranteed"
            print("[MAR MQTT] ERROR:", self.last_error)

    def emit_status(self, snapshot: dict):
        """Publish at most once per second; drop offline telemetry, never full landmarks."""
        if not self.enabled or not self.connected.is_set():
            return
        now = time.monotonic()
        if now - self._last_status < 1:
            return
        self._last_status = now
        sos = snapshot.get("sos") or {}
        doc = {
            "robot_id": self.robot_id,
            "source": "VISION_AI",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "fps": snapshot.get("fps"),
            "people_count": snapshot.get("people_count", 0),
            "body_points": len(snapshot.get("body") or []),
            "hands_detected": len(snapshot.get("hands") or []),
            "face_detected": bool(snapshot.get("face")),
            "fall_state": snapshot.get("fall_state"),
            "seizure_phase": (snapshot.get("seizure") or {}).get("phase"),
            "sos_phase": sos.get("phase"),
            "simulation": True,
        }
        # Non-blocking QoS0; not suitable for critical events.
        self._client.publish(
            self.status_topic, json.dumps(doc, ensure_ascii=False),
            qos=0, retain=False,
        )

    def _publish_worker(self):
        while not self._stop.is_set():
            if not self.connected.wait(timeout=0.5):
                continue
            try:
                payload = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                info = self._client.publish(
                    self.topic,
                    json.dumps(payload, ensure_ascii=False),
                    qos=1,
                    retain=False,
                )
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    raise RuntimeError(f"publish rc={info.rc}")
                self.sent_events += 1
                info.wait_for_publish(timeout=5)
                if not info.is_published():
                    raise TimeoutError("QoS1 PUBACK timeout")
                self.ack_events += 1
                print(f"[MAR MQTT] PUBACK event={payload['event_type']} id={payload['event_id']}")
            except Exception as exc:
                self.last_error = f"Publish failed: {exc}"
                print("[MAR MQTT]", self.last_error)
                # Requeue; Backend must deduplicate by event_id under QoS1.
                try:
                    self._pending.put_nowait(payload)
                except queue.Full:
                    print("[MAR MQTT] LOST EVENT due to full queue:", payload["event_id"])
                time.sleep(1)
            finally:
                self._pending.task_done()

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "connected": self.connected.is_set(),
            "event_topic": self.topic if self.enabled else None,
            "status_topic": self.status_topic if self.enabled else None,
            "pending_events": self._pending.qsize(),
            "published": self.sent_events,
            "acked": self.ack_events,
            "last_error": self.last_error,
        }

    def close(self):
        self._stop.set()
        if self._client is not None:
            self._client.disconnect()
            self._client.loop_stop()
        if self._worker is not None:
            self._worker.join(timeout=2)
