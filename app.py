
import os
import cv2
import math
import time
import uuid
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import mediapipe as mp
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, StreamingResponse

BASE = Path(__file__).resolve().parent
MODEL = str(BASE / "pose_landmarker.task")

lock = threading.Lock()
stop_event = threading.Event()
latest_jpeg = None

latest = {
    "state": "STARTING",
    "fps": 0,
    "landmarks": [],
    "metrics": {},
    "last_event": None
}

SKELETON = [
    # Dau va co
    (0, 11), (0, 12),
    # Than tren va tay
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    # Chan
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]


from fall_detector import FallDetector, MultiPersonFallTracker

PERSON_COLORS = [
    (74, 222, 128),   # Green
    (56, 189, 248),   # Sky blue
    (251, 146, 60),   # Orange
    (244, 114, 182),  # Pink
    (167, 139, 250),  # Purple
    (250, 204, 21),   # Yellow
]

def draw_pose(frame, points, label=None, color=None):
    h, w = frame.shape[:2]
    line_color = color if color is not None else (74, 222, 128)

    for a, b in SKELETON:
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        vis_a = getattr(pa, "visibility", 1.0)
        vis_b = getattr(pb, "visibility", 1.0)
        if (vis_a is not None and vis_a < 0.20) or (vis_b is not None and vis_b < 0.20):
            continue

        start = (int(pa.x * w), int(pa.y * h))
        end = (int(pb.x * w), int(pb.y * h))
        cv2.line(frame, start, end, line_color, 2)

    for p in points:
        vis = getattr(p, "visibility", 1.0)
        if vis is not None and vis >= 0.20:
            x = int(p.x * w)
            y = int(p.y * h)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    if label:
        ref_p = points[0] if len(points) > 0 else None
        if ref_p is not None:
            if getattr(ref_p, "visibility", 1.0) < 0.20 and len(points) > 12:
                s11_vis = getattr(points[11], "visibility", 0) or 0
                s12_vis = getattr(points[12], "visibility", 0) or 0
                if max(s11_vis, s12_vis) > 0.15:
                    ref_p = points[11] if s11_vis > s12_vis else points[12]
            nx = int(ref_p.x * w)
            ny = max(24, int(ref_p.y * h) - 14)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (nx - 4, ny - th - 4), (nx + tw + 4, ny + 4), (15, 23, 42), -1)
            cv2.putText(frame, label, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
def camera_worker():
    global latest_jpeg, latest

    detector = FallDetector()
    last_event = None
    previous_frame_time = None

    source_text = os.getenv("VIDEO_SOURCE", "0")
    source = int(source_text) if source_text.isdigit() else source_text

    cap = None

    try:
        if isinstance(source, int) and os.name == "nt":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError("Khong mo duoc camera/video")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=MODEL
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=int(os.getenv("MAX_POSES", "6")),
            min_pose_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as model:

            while not stop_event.is_set():
                ok, frame = cap.read()

                if not ok:
                    if not isinstance(source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok:
                        with lock:
                            latest["state"] = "VIDEO_ENDED_OR_CAMERA_ERROR"
                        break

                h_orig, w_orig = frame.shape[:2]
                if max(h_orig, w_orig) > 960:
                    scale = 960.0 / max(h_orig, w_orig)
                    frame = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

                now = time.monotonic()
                h, w = frame.shape[:2]

                fps = 0
                if previous_frame_time is not None:
                    dt = now - previous_frame_time
                    if dt > 0:
                        fps = round(1 / dt, 1)

                previous_frame_time = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = model.detect_for_video(
                    image,
                    int(now * 1000)
                )

                state = "NO_PERSON"
                metrics = {}
                landmarks = []

                if result.pose_landmarks:
                    pose = result.pose_landmarks[0]

                    world = (
                        result.pose_world_landmarks[0]
                        if result.pose_world_landmarks
                        else None
                    )

                    for i, p in enumerate(pose):
                        wp = world[i] if world else None

                        landmarks.append({
                            "id": i,
                            "x": round(float(p.x), 4),
                            "y": round(float(p.y), 4),
                            "z": round(float(p.z), 4),
                            "visibility": round(
                                float(p.visibility), 3
                            ),
                            "x_m": round(float(wp.x), 4) if wp else None,
                            "y_m": round(float(wp.y), 4) if wp else None,
                            "z_m": round(float(wp.z), 4) if wp else None
                        })

                    state, metrics, triggered = detector.update(
                        pose, w, h, now
                    )

                    draw_pose(frame, pose)

                    if triggered:
                        last_event = {
                            "id": str(uuid.uuid4()),
                            "event_type": "SUSPECTED_FALL",
                            "detected_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "algorithm": "heuristic_v2",
                            "confidence": metrics.get("confidence", 0.85),
                            "metrics": metrics.copy()
                        }

                        print("SUSPECTED FALL:", last_event)

                else:
                    detector.reset_pose()

                color = (
                    (0, 0, 255)
                    if state == "SUSPECTED_FALL"
                    else (0, 255, 0)
                )

                cv2.putText(
                    frame, state, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2
                )

                cv2.putText(
                    frame, f"FPS: {fps}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                encoded, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                if encoded:
                    with lock:
                        latest_jpeg = buffer.tobytes()
                        latest = {
                            "state": state,
                            "fps": fps,
                            "landmarks": landmarks,
                            "metrics": metrics,
                            "last_event": last_event
                        }

                time.sleep(0.01)

    except Exception as exc:
        print("AI ERROR:", exc)

        with lock:
            latest["state"] = "ERROR: " + str(exc)

    finally:
        if cap is not None:
            cap.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event.clear()
    worker = threading.Thread(
        target=camera_worker,
        daemon=True
    )
    worker.start()

    yield

    stop_event.set()
    worker.join(timeout=3)


app = FastAPI(
    title="MediAssistRover Fall Detection AI",
    lifespan=lifespan
)


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/api/pose")
def get_pose():
    with lock:
        return dict(latest)


def video_generator():
    while not stop_event.is_set():
        with lock:
            frame = latest_jpeg

        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame + b"\r\n"
            )

        time.sleep(0.07)


@app.get("/video")
def video():
    return StreamingResponse(
        video_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_pose(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            with lock:
                snapshot = dict(latest)

            await ws.send_json(snapshot)
            await asyncio.sleep(0.2)
    except Exception:
        pass
