"""Lightweight Camera Streamer for Raspberry Pi (Robot Edge Node).

Chay tren Raspberry Pi cua Robot de truyen luong video MJPEG/HTTP
ve cho Server Backend (noi chay full_ai.py) ma khong ton CPU/RAM tren Pi.

Yeu cau tren Raspberry Pi:
    pip install opencv-python-headless
Chay tren Pi:
    python3 pi_camera_streamer.py
Stream URL de khai bao tren BE (VIDEO_SOURCE):
    http://<IP_RASPBERRY_PI>:8090/video
"""
import json
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

CAMERA_SOURCE = os.getenv("PI_CAMERA_SOURCE", "0")
STREAM_PORT = int(os.getenv("PI_STREAM_PORT", "8090"))
FRAME_WIDTH = int(os.getenv("PI_FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("PI_FRAME_HEIGHT", "480"))
TARGET_FPS = max(1, int(os.getenv("PI_TARGET_FPS", "20")))
JPEG_QUALITY = max(30, min(95, int(os.getenv("PI_JPEG_QUALITY", "75"))))


class PiCameraBuffer:
    """Doc frame lien tuc tu camera trong thread rieng, luon giu frame moi nhat."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest_jpg = None
        self.frame_id = 0
        self.fps = 0.0
        self.status = "STARTING"
        self.stop_event = threading.Event()

    def update(self, jpg_bytes: bytes, fps: float):
        with self.cond:
            self.latest_jpg = jpg_bytes
            self.frame_id += 1
            self.fps = round(fps, 1)
            self.status = "STREAMING"
            self.cond.notify_all()

    def get_latest(self, last_id: int, timeout: float = 1.0):
        with self.cond:
            if self.frame_id == last_id:
                self.cond.wait(timeout=timeout)
            return self.frame_id, self.latest_jpg


buffer = PiCameraBuffer()


def open_pi_camera(source_raw: str):
    """Mo camera USB (/dev/video0) hoac CSI (GStreamer/libcamera) tren Raspberry Pi."""
    if source_raw.isdigit():
        idx = int(source_raw)
        if os.name == "nt":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
    elif source_raw.lower() == "csi":
        pipeline = (
            f"libcamerasrc ! video/x-raw, width={FRAME_WIDTH}, height={FRAME_HEIGHT}, "
            f"framerate={TARGET_FPS}/1 ! videoconvert ! appsink drop=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source_raw)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_loop():
    interval = 1.0 / TARGET_FPS
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    prev_time = time.monotonic()

    while not buffer.stop_event.is_set():
        cap = open_pi_camera(CAMERA_SOURCE)
        if not cap or not cap.isOpened():
            buffer.status = f"CAMERA_OFFLINE ({CAMERA_SOURCE})"
            print(f"[PI-CAMERA] Khong mo duoc camera '{CAMERA_SOURCE}', thu lai sau 2s...")
            time.sleep(2.0)
            continue

        print(f"[PI-CAMERA] Da ket noi camera '{CAMERA_SOURCE}' ({FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS}fps)")
        try:
            while not buffer.stop_event.is_set():
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[PI-CAMERA] Mat tin hieu frame, dang ket noi lai...")
                    break

                ok_enc, jpg = cv2.imencode(".jpg", frame, encode_params)
                if ok_enc:
                    now = time.monotonic()
                    dt = max(now - prev_time, 1e-4)
                    prev_time = now
                    buffer.update(jpg.tobytes(), 1.0 / dt)

                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        finally:
            cap.release()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # Tat log HTTP moi frame de giam tai I/O tren Raspberry Pi

    def do_GET(self):
        if self.path.startswith("/video") or self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_id = -1
            try:
                while not buffer.stop_event.is_set():
                    last_id, jpg = buffer.get_latest(last_id, timeout=1.0)
                    if not jpg:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path.startswith("/snapshot"):
            _, jpg = buffer.get_latest(-1, timeout=0.2)
            if not jpg:
                self.send_error(503, "Camera frame not ready")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path.startswith("/health") or self.path == "/":
            payload = json.dumps({
                "service": "MAR-Pi-Camera-Streamer",
                "status": buffer.status,
                "fps": buffer.fps,
                "frame_id": buffer.frame_id,
                "resolution": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                "stream_endpoint": "/video"
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404, "Not Found")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, name="PiCameraCapture", daemon=True)
    t.start()
    server = ThreadedHTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
    print(f"[PI-CAMERA] Server dang phat video tai: http://0.0.0.0:{STREAM_PORT}/video")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        buffer.stop_event.set()
        server.server_close()
