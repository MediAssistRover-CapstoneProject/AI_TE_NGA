"""Headless fall inference for stationary robot observation windows."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import hmac
from io import BytesIO
import os
from pathlib import Path
import threading
import time
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from fall_detector import MultiPersonFallTracker


ALGORITHM = "yolov8n-pose-fall-v2"
MAX_FRAME_BYTES = 256 * 1024
MAX_FRAME_EDGE = 640
MODEL_EDGE = 416
MAX_SESSIONS = 8
SESSION_IDLE_SECONDS = 30
MAX_FRAME_GAP_SECONDS = 1
# COCO pose indices mapped to the landmark indices FallDetector consumes.
LANDMARKS = {5: 11, 6: 12, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}


class FrameError(ValueError):
    pass


def decode_frame(data):
    try:
        with Image.open(BytesIO(data), formats=("JPEG",)) as source:
            width, height = source.size
            if not (32 <= width <= MAX_FRAME_EDGE and 32 <= height <= MAX_FRAME_EDGE):
                raise FrameError("INVALID_FRAME_DIMENSIONS")
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise FrameError("INVALID_JPEG") from None


class PoseModel:
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        metadata = self.session.get_modelmeta().custom_metadata_map
        if (len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4
                or metadata.get("task") != "pose" or metadata.get("kpt_shape") != "[17, 3]"
                or any(isinstance(actual, int) and actual != expected
                       for actual, expected in zip(inputs[0].shape, (1, 3, MODEL_EDGE, MODEL_EDGE)))):
            raise ValueError("Expected a raw YOLOv8n pose export with 17 keypoints")
        self.input_name = inputs[0].name

    def poses(self, frame):
        width, height = frame.size
        scale = MODEL_EDGE / max(width, height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        x = (MODEL_EDGE - resized_width) // 2
        y = (MODEL_EDGE - resized_height) // 2
        canvas = Image.new("RGB", (MODEL_EDGE, MODEL_EDGE), (114, 114, 114))
        canvas.paste(frame.resize((resized_width, resized_height), Image.Resampling.BILINEAR), (x, y))
        pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(pixels) / np.float32(255)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 1 or outputs[0].ndim != 3 or outputs[0].shape[:2] != (1, 56):
            raise RuntimeError("Unexpected YOLO pose output")
        rows = outputs[0][0].T
        rows = rows[(rows[:, 4] >= .4) & np.isfinite(rows).all(axis=1)
                    & (rows[:, 2] > 0) & (rows[:, 3] > 0)]
        if not len(rows):
            return []
        rows = rows[np.argsort(-rows[:, 4], kind="stable")[:128]]
        poses = []
        centers, half = rows[:, :2], rows[:, 2:4] / 2
        minimums, maximums = centers - half, centers + half
        areas = rows[:, 2] * rows[:, 3]
        remaining = np.arange(len(rows))
        while remaining.size and len(poses) < 8:
            index, remaining = remaining[0], remaining[1:]
            keypoints = rows[index, 5:].reshape(17, 3)
            if np.all(keypoints[[5, 6, 11, 12], 2] >= .5):
                pose = [{"x": 0., "y": 0., "visibility": 0.} for _ in range(33)]
                for coco, expected in LANDMARKS.items():
                    point_x, point_y, confidence = keypoints[coco]
                    pose[expected] = {"x": float(np.clip((point_x-x) / resized_width, 0, 1)),
                                      "y": float(np.clip((point_y-y) / resized_height, 0, 1)),
                                      "visibility": float(confidence)}
                poses.append(pose)
            overlap = np.maximum(0, np.minimum(maximums[index], maximums[remaining])
                                 - np.maximum(minimums[index], minimums[remaining])).prod(axis=1)
            remaining = remaining[overlap / (areas[index] + areas[remaining] - overlap) < .45]
        return poses


@dataclass
class Observation:
    tracker: MultiPersonFallTracker
    received_at: float
    digest: bytes = b""
    confidence: float | None = None


class FallService:
    def __init__(self, model):
        self.model = model
        self.sessions = {}
        self.inference_lock = threading.Lock()

    def prune(self, now):
        self.sessions = {key: state for key, state in self.sessions.items()
                         if now - state.received_at <= SESSION_IDLE_SECONDS}

    def observe(self, session, data, received_at):
        if not self.inference_lock.acquire(blocking=False):
            raise HTTPException(429, "FALL_BUSY")
        try:
            self.prune(received_at)
            state = self.sessions.get(session)
            if state is None and len(self.sessions) >= MAX_SESSIONS:
                raise HTTPException(429, "FALL_SESSION_LIMIT")
            digest = hashlib.sha256(data).digest()
            if state is not None and received_at - state.received_at <= MAX_FRAME_GAP_SECONDS:
                if digest == state.digest:
                    return result(state.confidence)
            else:
                state = Observation(MultiPersonFallTracker(max_distance=.25), received_at)

            with decode_frame(data) as frame:
                width, height = frame.size
                poses = self.model.poses(frame)
            _, _, triggered, info = state.tracker.update(poses, width, height, received_at)
            # A track missing from this frame cannot complete a prior transition.
            for track in state.tracker.tracks.values():
                if track["seen"] != received_at:
                    track["detector"].reset_pose()
            state.received_at = received_at
            state.digest = digest
            if triggered and info is not None:
                state.confidence = info["metrics"]["confidence"]
            self.sessions[session] = state
            return result(state.confidence)
        finally:
            self.inference_lock.release()

    async def expire(self):
        while True:
            await asyncio.sleep(5)
            if self.inference_lock.acquire(blocking=False):
                try:
                    self.prune(time.monotonic())
                finally:
                    self.inference_lock.release()


def result(confidence):
    return {"suspected_fall": confidence is not None, "confidence": confidence, "algorithm": ALGORITHM}


@asynccontextmanager
async def lifespan(app):
    token = os.environ.get("FALL_API_TOKEN", "")
    if len(token) < 32 or any(char.isspace() for char in token) or not token.isascii():
        raise RuntimeError("FALL_API_TOKEN must contain at least 32 ASCII characters without whitespace")
    path = os.environ.get("FALL_MODEL_PATH", Path(__file__).with_name("yolov8n-pose.onnx"))
    app.state.service = FallService(PoseModel(path))
    app.state.authorization = "Bearer " + token
    app.state.receiving = asyncio.Lock()
    cleanup = asyncio.create_task(app.state.service.expire())
    try:
        yield
    finally:
        cleanup.cancel()
        await asyncio.gather(cleanup, return_exceptions=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
@app.get("/v1/fall/health")
def health():
    return {"state": "READY", "algorithm": ALGORITHM}


@app.post("/v1/fall/frame")
async def frame(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.isascii() or not hmac.compare_digest(auth, app.state.authorization):
        raise HTTPException(401, "FALL_AUTH_REQUIRED")
    try:
        session = UUID(request.headers.get("X-Fall-Session", ""))
    except ValueError:
        raise HTTPException(422, "INVALID_FALL_SESSION") from None
    if request.headers.get("Content-Type", "").lower().split(";", 1)[0].strip() != "image/jpeg":
        raise HTTPException(415, "FALL_REQUIRES_JPEG")
    if app.state.receiving.locked():
        raise HTTPException(429, "FALL_BUSY")
    async with app.state.receiving:
        raw = bytearray()
        try:
            async with asyncio.timeout(2):
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > MAX_FRAME_BYTES:
                        raise HTTPException(413, "FALL_FRAME_TOO_LARGE")
                    raw.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, "FALL_FRAME_TIMEOUT") from None
        try:
            value = await run_in_threadpool(app.state.service.observe, session, raw, time.monotonic())
        except FrameError as error:
            raise HTTPException(422, str(error)) from None
    return JSONResponse(value, headers={"Cache-Control": "no-store"})
