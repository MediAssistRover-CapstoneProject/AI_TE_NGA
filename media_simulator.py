"""Use the laptop AI stream as a virtual robot camera for SOS calls."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import requests
from dotenv import load_dotenv
from livekit import rtc


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("media-simulator")
OPEN_STATES = {"RINGING", "CONNECTING", "CONNECTED", "RECONNECTING"}


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


class DeviceApi:
    def __init__(self) -> None:
        self.base_url = required("BE_BASE_URL").rstrip("/")
        self.robot_code = required("ROBOT_CODE")
        self.device_secret = required("ROBOT_DEVICE_SECRET")
        self.session = requests.Session()
        self._token = ""
        self._lock = threading.Lock()

    def authenticate(self) -> None:
        response = self.session.post(
            f"{self.base_url}/api/v1/pi/auth/token",
            json={"robotCode": self.robot_code, "deviceSecret": self.device_secret},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        with self._lock:
            self._token = payload["accessToken"]
        LOG.info("DEVICE_AUTHENTICATED robot_id=%s", payload["robotId"])

    def headers(self) -> dict[str, str]:
        with self._lock:
            return {"Authorization": f"Bearer {self._token}"}

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        for attempt in range(2):
            extra_headers = kwargs.pop("headers", {}) or {}
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers={**extra_headers, **self.headers()},
                timeout=15,
                **kwargs,
            )
            if response.status_code != 401 or attempt:
                response.raise_for_status()
                return response
            self.authenticate()
        raise RuntimeError("DEVICE_AUTH_FAILED")

    def current(self) -> dict[str, Any]:
        return self.request("GET", "/api/v1/pi/media-session").json()

    def credentials(self, session_id: str) -> dict[str, Any]:
        return self.request(
            "POST", f"/api/v1/pi/media-sessions/{session_id}/join-token"
        ).json()

    def report(self, session_id: str, state: str, reason: str | None = None) -> None:
        body: dict[str, str] = {"state": state}
        if reason:
            body["reason"] = reason
        self.request(
            "POST", f"/api/v1/pi/media-sessions/{session_id}/state", json=body
        )

    def listen(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                with self.session.get(
                    f"{self.base_url}/api/v1/pi/media-session/events",
                    headers={**self.headers(), "Accept": "text/event-stream"},
                    stream=True,
                    timeout=(15, 70),
                ) as response:
                    if response.status_code == 401:
                        self.authenticate()
                        continue
                    response.raise_for_status()
                    LOG.info("MEDIA_SSE_READY - BE can now open calls for this robot")
                    for _line in response.iter_lines():
                        if stop.is_set():
                            return
            except Exception as exc:
                if not stop.is_set():
                    LOG.warning("MEDIA_SSE_RECONNECTING error=%s", type(exc).__name__)
                    stop.wait(2)


class CameraPublisher:
    def __init__(self, source_url: str) -> None:
        self.source_url = source_url

    async def run(self, api: DeviceApi, session: dict[str, Any]) -> None:
        session_id = str(session["sessionId"])
        credentials = await asyncio.to_thread(api.credentials, session_id)
        width = int(credentials["width"])
        height = int(credentials["height"])
        fps = int(credentials["fps"])
        room = rtc.Room()
        capture: cv2.VideoCapture | None = None
        try:
            await asyncio.wait_for(
                room.connect(
                    credentials["url"],
                    credentials["token"],
                    options=rtc.RoomOptions(auto_subscribe=True),
                ),
                timeout=20,
            )
            video = rtc.VideoSource(width, height)
            track = rtc.LocalVideoTrack.create_video_track("robot-camera", video)
            await room.local_participant.publish_track(
                track,
                rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_CAMERA,
                    video_codec=rtc.VideoCodec.VP8,
                    simulcast=False,
                    video_encoding=rtc.VideoEncoding(
                        max_bitrate=int(credentials["bitrate"]), max_framerate=fps
                    ),
                ),
            )
            capture = cv2.VideoCapture(self.source_url)
            if not capture.isOpened():
                raise RuntimeError("AI_VIDEO_STREAM_NOT_AVAILABLE")
            await asyncio.to_thread(api.report, session_id, "CONNECTED")
            LOG.info("CALL_CONNECTED session=%s - camera is streaming to staff", session_id)

            frame_period = 1 / max(fps, 1)
            next_report = time.monotonic() + 10
            while True:
                started = time.monotonic()
                ok, pixels = await asyncio.to_thread(capture.read)
                if not ok or pixels is None:
                    raise RuntimeError("AI_VIDEO_STREAM_LOST")
                if pixels.shape[:2] != (height, width):
                    pixels = cv2.resize(pixels, (width, height))
                yuv = cv2.cvtColor(pixels, cv2.COLOR_BGR2YUV_I420)
                video.capture_frame(
                    rtc.VideoFrame(
                        width, height, rtc.VideoBufferType.I420, memoryview(yuv).cast("B")
                    )
                )
                now = time.monotonic()
                if now >= next_report:
                    current = await asyncio.to_thread(api.current)
                    active = current.get("session")
                    if (
                        not active
                        or str(active.get("sessionId")) != session_id
                        or active.get("state") not in OPEN_STATES
                    ):
                        LOG.info("CALL_ENDED session=%s", session_id)
                        return
                    await asyncio.to_thread(api.report, session_id, "CONNECTED")
                    next_report = now + 10
                await asyncio.sleep(max(0, frame_period - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.error("CALL_FAILED session=%s error=%s", session_id, exc)
            try:
                await asyncio.to_thread(
                    api.report, session_id, "FAILED", "MEDIA_SIMULATOR_FAILED"
                )
            except Exception:
                LOG.exception("Could not report media failure to BE")
        finally:
            if capture is not None:
                capture.release()
            await room.disconnect()


async def main() -> None:
    api = DeviceApi()
    source_url = required("AI_VIDEO_URL")
    await asyncio.to_thread(api.authenticate)
    stop = threading.Event()
    threading.Thread(target=api.listen, args=(stop,), daemon=True).start()
    publisher = CameraPublisher(source_url)
    active_id: str | None = None

    LOG.info("MEDIA_SIMULATOR_STARTED video=%s", source_url)
    LOG.info("Keep this window open, then trigger SOS from the AI camera")
    try:
        while True:
            try:
                current = await asyncio.to_thread(api.current)
                session = current.get("session") if current.get("enabled") else None
                session_id = str(session["sessionId"]) if session else None
                # RINGING means the assigned staff has not accepted yet. The
                # device join-token intentionally returns 422 in that state.
                # Wait for CONNECTING instead of consuming the session once.
                if (
                    session
                    and session_id != active_id
                    and session.get("state") in {"CONNECTING", "CONNECTED", "RECONNECTING"}
                ):
                    active_id = session_id
                    try:
                        await publisher.run(api, session)
                    finally:
                        active_id = None
            except requests.RequestException as exc:
                LOG.warning("BE_POLL_FAILED error=%s", exc)
            await asyncio.sleep(2)
    finally:
        stop.set()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info("MEDIA_SIMULATOR_STOPPED")
