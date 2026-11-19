"""RGB pose suspicions; clients verify static floor postures with depth."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
from io import BytesIO
import logging
import math
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
from starlette.requests import ClientDisconnect

from fall_detector import (BODY_JOINTS, MultiPersonFallTracker, center_point,
                           visible_hips, visible_image_point)


ALGORITHM = "yolov8n-pose-fall-v5"
MAX_FRAME_BYTES = 256 * 1024
MAX_FRAME_EDGE = 640
MODEL_EDGE = 416
MAX_SESSIONS = 8
SESSION_IDLE_SECONDS = 30
MAX_FRAME_GAP_SECONDS = 1
MIN_SEPARATION_NORMALIZED = .12
LOGGER = logging.getLogger("fall_ai")
# COCO pose indices mapped to the landmark indices FallDetector consumes.
LANDMARKS = {5: 11, 6: 12, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}


class FrameError(ValueError):
    pass


class PoseBatch(list):
    """Keep detector boxes beside the unchanged poses used by fall/depth logic."""
    def __init__(self):
        super().__init__()
        self.detections = []


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
            return PoseBatch()
        rows = rows[np.argsort(-rows[:, 4], kind="stable")[:128]]
        poses = PoseBatch()
        centers, half = rows[:, :2], rows[:, 2:4] / 2
        minimums, maximums = centers - half, centers + half
        areas = rows[:, 2] * rows[:, 3]
        remaining = np.arange(len(rows))
        while remaining.size and len(poses.detections) < 8:
            index, remaining = remaining[0], remaining[1:]
            pose_index = None
            keypoints = rows[index, 5:].reshape(17, 3)
            if np.any(keypoints[[11, 12], 2] >= .5):
                pose = [{"x": 0., "y": 0., "visibility": 0.} for _ in range(33)]
                for coco, expected in LANDMARKS.items():
                    point_x, point_y, confidence = keypoints[coco]
                    # Undo letterbox padding without clipping points into the image.
                    pose[expected] = {"x": float((point_x - x) / resized_width),
                                      "y": float((point_y - y) / resized_height),
                                      "visibility": float(confidence)}
                if visible_hips(pose):
                    pose_index = len(poses)
                    poses.append(pose)
            left, top = np.maximum(0., (minimums[index] - (x, y)) / (resized_width, resized_height))
            right, bottom = np.minimum(1., (maximums[index] - (x, y)) / (resized_width, resized_height))
            if right > left and bottom > top:
                poses.detections.append({"pose_index": pose_index,
                    "bbox": [float(left), float(top), float(right-left), float(bottom-top)],
                    "confidence": float(np.clip(rows[index, 4], 0., 1.))})
            overlap = np.maximum(0, np.minimum(maximums[index], maximums[remaining])
                                 - np.maximum(minimums[index], minimums[remaining])).prod(axis=1)
            remaining = remaining[overlap / (areas[index] + areas[remaining] - overlap) < .45]
        return poses


@dataclass
class Observation:
    tracker: MultiPersonFallTracker
    received_at: float
    frame_size: tuple[int, int]
    digest: bytes = b""
    confidence: float | None = None
    person: dict | None = None
    detection_kind: str | None = None
    visual_review_eligible: bool = False
    sample_at: float = 0.
    capture_timed: bool = False
    camera_moving: bool = False
    overlay: dict | None = None
    floor_candidates: dict = field(default_factory=dict)
    floor_candidate: dict | None = None


def overlay_result(poses, tracks, width, height, captured_at, moving):
    by_pose = {track["pose_index"]: track for track in tracks}
    detections = []
    for detection in getattr(poses, "detections", [])[:8]:
        track = by_pose.get(detection["pose_index"])
        state = "PERSON"
        if track:
            state = {"SUSPECTED_FALL": "SUSPECTED_FALL",
                     "SUSPECTED_PERSON_ON_FLOOR": "PERSON_ON_FLOOR",
                     "CHECKING": "OBSERVING", "HORIZONTAL_POSTURE": "OBSERVING"}.get(track["state"], "PERSON")
        detections.append({"track_id": track["person_id"] if track else None,
                           "bbox": detection["bbox"], "confidence": detection["confidence"], "state": state})
    stamp = (datetime.fromtimestamp(captured_at, timezone.utc).isoformat()
             if captured_at is not None else None)
    return {"width": width, "height": height, "captured_at": stamp,
            "camera_moving": moving, "detections": detections}


def pose_points(pose, joints):
    return [visible_image_point(pose[i]) if i < len(pose) else None for i in joints]


def person_geometry(poses, pose_index, width, height, depth_joints):
    points = pose_points(poses[pose_index], depth_joints)
    if any(point is None for point in points):
        return None
    hip_center = center_point(visible_hips(poses[pose_index]).values())
    for index, pose in enumerate(poses):
        if index == pose_index:
            continue
        other_hips = visible_hips(pose)
        body = [point for point in pose_points(pose, BODY_JOINTS) if point is not None]
        xs, ys = zip(*body)
        if any(min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys) for x, y in points):
            return None
        # Nearby tracks can be confused during matching or depth sampling.
        if math.dist(hip_center, center_point(other_hips.values())) < MIN_SEPARATION_NORMALIZED:
            return None
    for box in getattr(poses, "detections", []):
        if box["pose_index"] is not None:
            continue
        x, y, box_width, box_height = box["bbox"]
        if any(x <= px <= x + box_width and y <= py <= y + box_height for px, py in points):
            return None
    visible = [point for point in pose_points(poses[pose_index], BODY_JOINTS) if point is not None]
    xs, ys = zip(*visible)
    box = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
    for detection in getattr(poses, "detections", []):
        if detection["pose_index"] == pose_index:
            box = detection["bbox"]
            break
    roles = {23: 'hip', 24: 'hip', 25: 'knee', 26: 'knee', 27: 'ankle', 28: 'ankle',
             11: 'shoulder', 12: 'shoulder'}
    return {"image_width": width, "image_height": height, "bbox": box,
            "keypoint_roles": [roles.get(joint, 'body') for joint in depth_joints],
            "keypoints": [{"x": x, "y": y} for x, y in points]}


def depth_floor_candidate(state, poses, tracks, width, height, now, moving):
    """An end-on image is ambiguous; only depth can establish a low 3-D body.

    This does not set suspected_fall or bypass staff camera review. Keep a
    continuous, bounded view of the same track before asking Pi to measure it.
    """
    current = {}
    ready = []
    for track in tracks if not moving and os.getenv('FLOOR_END_ON_ENABLED', 'true').lower() == 'true' else ():
        if track['state'] == 'AMBIGUOUS_PERSON':
            continue
        index = track['pose_index']
        pose = poses[index]
        choices = []
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            first = visible_image_point(pose[hip]) if hip < len(pose) else None
            if first is None or first[1] <= .45:
                continue
            for support in (ankle, knee):
                second = visible_image_point(pose[support]) if support < len(pose) else None
                if second is not None:
                    length = math.hypot((first[0]-second[0])*width, (first[1]-second[1])*height)
                    if length >= 12.:
                        choices.append((support in (27, 28), length, (hip, support)))
        if not choices:
            continue
        joints = max(choices)[2]
        person = person_geometry(poses, index, width, height, joints)
        if person is None:
            continue
        tid = track['person_id']
        prior = state.floor_candidates.get(tid)
        center = track['center']
        continuous = (prior is not None and 0. < now-prior['last'] <= 1.
                      and math.dist(center, prior['center']) <= .08)
        since = prior['since'] if continuous else now
        current[tid] = {'since': since, 'last': now, 'center': center}
        detector = state.tracker.tracks[tid]['detector']
        if detector.floor_enabled and now-since >= detector.floor_confirm_seconds:
            ready.append({'reason': 'END_ON_LOWER_BODY', 'confidence': .55, 'track_id': tid,
                          'ground_confirmation_required': True, 'visual_review_eligible': False,
                          'person': person})
    state.floor_candidates = current
    return ready[0] if len(ready) == 1 else None


class FallService:
    def __init__(self, model):
        self.model = model
        self.sessions = {}
        self.inference_lock = threading.Lock()

    def prune(self, now):
        self.sessions = {key: state for key, state in self.sessions.items()
                         if now - state.received_at <= SESSION_IDLE_SECONDS}

    def observe(self, session, data, received_at, captured_at=None, camera_moving=False):
        if not self.inference_lock.acquire(blocking=False):
            raise HTTPException(429, "FALL_BUSY")
        try:
            self.prune(received_at)
            state = self.sessions.get(session)
            if state is None and len(self.sessions) >= MAX_SESSIONS:
                raise HTTPException(429, "FALL_SESSION_LIMIT")
            digest = hashlib.sha256(data).digest()
            sample_at = received_at if captured_at is None else captured_at
            capture_timed = captured_at is not None
            if (state is not None and received_at - state.received_at <= MAX_FRAME_GAP_SECONDS
                    and digest == state.digest and state.capture_timed == capture_timed
                    and state.camera_moving == camera_moving
                    and (not capture_timed or sample_at == state.sample_at)):
                return result(state.confidence, state.person, state.detection_kind, state.overlay,
                              state.visual_review_eligible, state.floor_candidate)
            if (state is not None and capture_timed and state.capture_timed
                    and sample_at <= state.sample_at):
                raise FrameError("FALL_FRAME_OUT_OF_ORDER")

            with decode_frame(data) as frame:
                width, height = frame.size
                poses = self.model.poses(frame)
            if (state is None or received_at - state.received_at > MAX_FRAME_GAP_SECONDS
                    or state.frame_size != (width, height) or state.capture_timed != capture_timed
                    or state.camera_moving != camera_moving or sample_at - state.sample_at > MAX_FRAME_GAP_SECONDS):
                state = Observation(MultiPersonFallTracker(max_distance=.25, allow_partial=True),
                                    received_at, (width, height),
                                    sample_at=sample_at, capture_timed=capture_timed, camera_moving=camera_moving)
            tracks, _, _, _ = state.tracker.update(poses, width, height, sample_at,
                                                 allow_transition=not camera_moving)
            state.overlay = overlay_result(poses, tracks, width, height, captured_at, camera_moving)
            state.floor_candidate = depth_floor_candidate(state, poses, tracks, width, height, sample_at, camera_moving)
            triggers = [track for track in tracks if track["triggered"]]
            # A track missing from this frame cannot complete a prior transition.
            for track in state.tracker.tracks.values():
                if track["seen"] != sample_at:
                    track["detector"].reset_pose()
            state.received_at = received_at
            state.digest = digest
            state.sample_at = sample_at
            state.person = None
            state.visual_review_eligible = False
            # A static-posture suspicion is current-frame evidence, not a
            # sticky event that can be attached to another person's later image.
            transition_visible = any(track["state"] == "SUSPECTED_FALL" for track in tracks)
            if state.detection_kind == "PERSON_ON_FLOOR" or (
                    state.detection_kind == "FALL_TRANSITION" and not transition_visible):
                state.confidence, state.detection_kind = None, None
            floor_tracks = [track for track in tracks if track["state"] == "SUSPECTED_PERSON_ON_FLOOR"]
            if triggers:
                scores = [track["metrics"]["confidence"] for track in triggers
                          if track["metrics"]["confidence"] is not None]
                if scores:
                    state.confidence = max(scores)
                    kinds = {track["metrics"].get("detection_kind", "FALL_TRANSITION") for track in triggers}
                    state.detection_kind = "PERSON_ON_FLOOR" if kinds == {"PERSON_ON_FLOOR"} else "FALL_TRANSITION"
                if len(triggers) == 1 and scores:
                    # Geometry is valid only for that person's actual trigger frame.
                    trigger = triggers[0]
                    metrics = trigger["metrics"]
                    state.person = person_geometry(poses, trigger["pose_index"], width, height,
                                                   metrics["depth_joints"])
                    state.visual_review_eligible = metrics["visual_review_eligible"] and state.person is not None
                    LOGGER.warning("FALL_RGB_SUSPICION kind=%s track_id=%d body_model=%s depth_points=%s "
                                   "visual_review_eligible=%s", state.detection_kind, trigger["person_id"],
                                   metrics["body_model"], state.person is not None, state.visual_review_eligible)
            elif floor_tracks and state.detection_kind != "FALL_TRANSITION":
                state.confidence = max(track["metrics"]["confidence"] for track in floor_tracks)
                state.detection_kind = "PERSON_ON_FLOOR"
                if len(floor_tracks) == 1:
                    track = floor_tracks[0]
                    metrics = track["metrics"]
                    state.person = person_geometry(poses, track["pose_index"], width, height,
                                                   metrics["depth_joints"])
                    state.visual_review_eligible = metrics["visual_review_eligible"] and state.person is not None
            self.sessions[session] = state
            return result(state.confidence, state.person, state.detection_kind, state.overlay,
                          state.visual_review_eligible, state.floor_candidate)
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


def result(confidence, person=None, detection_kind=None, overlay=None, visual_review_eligible=False, floor_candidate=None):
    value = {"suspected_fall": confidence is not None, "confidence": confidence, "algorithm": ALGORITHM}
    if confidence is not None:
        value["detection_kind"] = detection_kind or "FALL_TRANSITION"
        value["ground_confirmation_required"] = detection_kind == "PERSON_ON_FLOOR"
        if detection_kind == "PERSON_ON_FLOOR":
            value["visual_review_eligible"] = visual_review_eligible
    if person is not None:
        value["person"] = person
    if overlay is not None:
        value["overlay"] = overlay
    if floor_candidate is not None and confidence is None:
        value["floor_candidate"] = floor_candidate
    return value


def frame_context(headers, now=None):
    """Acquisition time drives kinetics; receipt time is only for server expiry."""
    raw = headers.get("X-Fall-Captured-At")
    moving = headers.get("X-Fall-Camera-Moving")
    if moving is not None and moving not in ("true", "false"):
        raise FrameError("INVALID_FALL_CAMERA_MOTION")
    if raw is None:
        if moving is not None:
            raise FrameError("FALL_MOTION_REQUIRES_CAPTURE_TIME")
        return None, False
    try:
        if not 16 <= len(raw) <= 40:
            raise ValueError()
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            raise ValueError()
        captured_at = stamp.timestamp()
    except (ValueError, OverflowError):
        raise FrameError("INVALID_FALL_CAPTURE_TIME") from None
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    # Capture order and inter-frame gaps are checked within the camera stream.
    # The wall-clock check is only a bound on skew or a queued old frame.
    if not math.isfinite(captured_at) or not -2. <= now - captured_at <= 5.:
        raise FrameError("FALL_CAPTURE_TIME_NOT_FRESH")
    return captured_at, moving == "true"


@asynccontextmanager
async def lifespan(app):
    token = os.environ.get("FALL_API_TOKEN", "")
    if len(token) < 32 or any(char.isspace() for char in token) or not token.isascii():
        raise RuntimeError("FALL_API_TOKEN must contain at least 32 ASCII characters without whitespace")
    path = os.environ.get("FALL_MODEL_PATH", Path(__file__).with_name("yolov8n-pose.onnx"))
    app.state.service = FallService(PoseModel(path))
    app.state.authorization = "Bearer " + token
    app.state.receiving = asyncio.Lock()
    app.state.last_frame_rejection = {}
    cleanup = asyncio.create_task(app.state.service.expire())
    try:
        yield
    finally:
        cleanup.cancel()
        await asyncio.gather(cleanup, return_exceptions=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def frame_rejected(error):
    code = str(error)
    now = time.monotonic()
    last = app.state.last_frame_rejection
    if now - last.get(code, -math.inf) >= 15:
        LOGGER.warning("FALL_FRAME_REJECTED code=%s", code)
        last[code] = now
    return HTTPException(422, code)


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
    try:
        captured_at, camera_moving = frame_context(request.headers)
    except FrameError as error:
        raise frame_rejected(error) from None
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
        except ClientDisconnect:
            # A timed-out/mobile client can leave during upload. This is a
            # request interruption, not an inference crash or a traceback.
            raise HTTPException(499, "FALL_CLIENT_DISCONNECTED") from None
        try:
            value = await run_in_threadpool(app.state.service.observe, session, raw, time.monotonic(),
                                           captured_at, camera_moving)
        except FrameError as error:
            raise frame_rejected(error) from None
    return JSONResponse(value, headers={"Cache-Control": "no-store"})
