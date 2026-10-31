import os
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "4")

import cv2
import math
import time
import uuid
import asyncio
import threading
import subprocess
import numpy as np

from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import mediapipe as mp
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from fall_detector import FallDetector, MultiPersonFallTracker
from app import draw_pose
from seizure_motion import SeizureMotionDetector

try:
    import torch
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    torch = None
    HAVE_YOLO = False

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = None

COCO_TO_MP = {
    0: 0,   # nose
    1: 2,   # left eye
    2: 5,   # right eye
    3: 7,   # left ear
    4: 8,   # right ear
    5: 11,  # left shoulder
    6: 12,  # right shoulder
    7: 13,  # left elbow
    8: 14,  # right elbow
    9: 15,  # left wrist
    10: 16, # right wrist
    11: 23, # left hip
    12: 24, # right hip
    13: 25, # left knee
    14: 26, # right knee
    15: 27, # left ankle
    16: 28, # right ankle
}


class PseudoLandmark:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def yolo_to_mp_landmarks(kpts_xy, kpts_conf, img_w, img_h):
    landmarks = [PseudoLandmark() for _ in range(33)]
    for coco_idx, mp_idx in COCO_TO_MP.items():
        if coco_idx < len(kpts_xy):
            kx, ky = kpts_xy[coco_idx]
            conf = float(kpts_conf[coco_idx]) if kpts_conf is not None else 0.8
            landmarks[mp_idx] = PseudoLandmark(
                x=float(kx) / max(img_w, 1),
                y=float(ky) / max(img_h, 1),
                z=0.0,
                visibility=conf
            )
    return landmarks


from mar_mqtt import VisionMqttPublisher
from hand_sos import HandSOSDetector

BASE = Path(__file__).resolve().parent

FACE_EVERY = max(1, int(os.getenv("FACE_EVERY", "6")))
HAND_EVERY = max(1, int(os.getenv("HAND_EVERY", "6")))
AI_EVERY = max(1, int(os.getenv("AI_EVERY", "2")))
ENABLE_FACE = os.getenv("ENABLE_FACE", "false").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_HAND = os.getenv("ENABLE_HAND", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_FRAME_DIM = max(320, int(os.getenv("MAX_FRAME_DIM", "800")))
TARGET_STREAM_FPS = max(10, min(30, int(os.getenv("TARGET_STREAM_FPS", "22"))))

lock = threading.Lock()
stop = threading.Event()
latest_jpg = None
jpg_event = threading.Event()

state = {
    "status": "STARTING",
    "fps": 0,
    "ai_fps": 0,
    "body": [],
    "hands": [],
    "face": None,
    "fall_state": "NO_DATA",
    "fall_metrics": {},
    "last_event": None,
}

PERSON_COLORS = [
    (74, 222, 128),   # Green
    (56, 189, 248),   # Sky blue
    (251, 146, 60),   # Orange
    (244, 114, 182),  # Pink
    (167, 139, 250),  # Purple
    (250, 204, 21),   # Yellow
]


class FastVideoReader:
    """Bo doc video toc do cao:
    - Voi file video 4K/FullHD: Su dung FFmpeg GPU Hardware Decoder (d3d11va + discard noref)
      de giai ma & scale ve 800p @ 22fps theo dung toc do 1.0x thoi gian thuc.
    - Voi Camera USB / Raspi HTTP Stream: Dung luong doc ngam xoa buffer tre hinh (0ms lag).
    """

    def __init__(self, source):
        self.source = source
        self.is_file = isinstance(source, str) and not source.lower().startswith(
            ("http://", "https://", "rtsp://", "rtmp://", "/dev/video")
        )
        self.proc = None
        self.cap = None
        self.out_w = 0
        self.out_h = 0
        self.frame_bytes = 0
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None

        if self.is_file and FFMPEG_BIN and os.path.exists(source):
            probe = cv2.VideoCapture(source)
            ok_p, sample_frame = probe.read()
            if ok_p and sample_frame is not None:
                h, w = sample_frame.shape[:2]
            else:
                w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe.release()
            if w > 0 and h > 0:
                scale = min(1.0, float(MAX_FRAME_DIM) / max(w, h))
                self.out_w = max(2, (int(w * scale) // 2) * 2)
                self.out_h = max(2, (int(h * scale) // 2) * 2)
                self.frame_bytes = self.out_w * self.out_h * 3
                self._start_ffmpeg()
                return

        self._open_cv_capture()
        if not self.is_file and self.cap and self.cap.isOpened():
            self.running = True
            self.thread = threading.Thread(target=self._live_reader_loop, daemon=True)
            self.thread.start()

    def _start_ffmpeg(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        cmd = [
            FFMPEG_BIN,
            "-hwaccel", "auto",
            "-discard", "noref",
            "-stream_loop", "-1",
            "-i", str(self.source),
            "-vf", f"fps={TARGET_STREAM_FPS},scale={self.out_w}:{self.out_h}:flags=fast_bilinear",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-loglevel", "error",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            bufsize=self.frame_bytes * 6
        )

    def _open_cv_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if isinstance(self.source, int) and os.name == "nt":
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _live_reader_loop(self):
        while self.running and not stop.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._open_cv_capture()
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.5)
                self._open_cv_capture()
                continue
            with self.frame_lock:
                self.latest_frame = frame

    def isOpened(self):
        return (self.proc is not None) or (self.cap is not None and self.cap.isOpened())

    def read(self):
        if self.proc is not None:
            raw = self.proc.stdout.read(self.frame_bytes)
            if len(raw) < self.frame_bytes:
                self._start_ffmpeg()
                raw = self.proc.stdout.read(self.frame_bytes)
                if len(raw) < self.frame_bytes:
                    return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.out_h, self.out_w, 3)).copy()
            return True, frame

        if not self.is_file:
            for _ in range(40):
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame = self.latest_frame.copy()
                        self.latest_frame = None
                        return True, frame
                if stop.is_set():
                    return False, None
                time.sleep(0.01)
            return False, None

        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.running = False
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FACE_LINES = [
    [10, 338, 297, 332, 284, 251, 389, 356,
     454, 323, 361, 288, 397, 365, 379, 378,
     400, 377, 152, 148, 176, 149, 150, 136,
     172, 58, 132, 93, 234, 127, 162, 21,
     54, 103, 67, 109, 10],
    [33, 160, 158, 133, 153, 144, 33],
    [362, 385, 387, 263, 373, 380, 362],
    [61, 40, 37, 0, 267, 270, 291,
     321, 314, 17, 84, 91, 61],
    [70, 63, 105, 66, 107],
    [336, 296, 334, 293, 300]
]

FINGER_JOINTS = {
    "thumb": (1, 2, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20)
}


def serialize(points, world=None):
    result = []
    for i, p in enumerate(points):
        item = {
            "id": i,
            "x": round(float(p.x), 4),
            "y": round(float(p.y), 4),
            "z": round(float(p.z), 4)
        }
        if world is not None:
            item["world"] = {
                "x": round(float(world[i].x), 4),
                "y": round(float(world[i].y), 4),
                "z": round(float(world[i].z), 4)
            }
        visibility = getattr(p, "visibility", None)
        if visibility is not None:
            item["visibility"] = round(float(visibility), 3)
        result.append(item)
    return result


def pixel(p, w, h):
    return (int(p.x * w), int(p.y * h))


def joint_angle(a, b, c, w, h):
    ax, ay = a.x * w - b.x * w, a.y * h - b.y * h
    cx, cy = c.x * w - b.x * w, c.y * h - b.y * h

    dot = ax * cx + ay * cy
    length = math.hypot(ax, ay) * math.hypot(cx, cy)

    if length < 1e-6:
        return 0

    cosine = max(-1, min(1, dot / length))
    return math.degrees(math.acos(cosine))


def fingers_of(hand, w, h):
    fingers = {}

    for name, (a, b, c) in FINGER_JOINTS.items():
        angle = joint_angle(hand[a], hand[b], hand[c], w, h)

        if angle > 155:
            finger_state = "OPEN"
        elif angle < 130:
            finger_state = "BENT"
        else:
            finger_state = "PARTIAL"

        fingers[name] = {
            "state": finger_state,
            "angle": round(angle, 1)
        }

    opened = [
        name for name, v in fingers.items()
        if v["state"] == "OPEN"
    ]

    if len(opened) == 5:
        gesture = "OPEN_PALM"
    elif len(opened) == 0:
        gesture = "CLOSED_HAND"
    elif set(opened) == {"index", "middle"}:
        gesture = "TWO_FINGERS"
    elif set(opened) == {"index"}:
        gesture = "POINTING"
    else:
        gesture = "FINGER_COMBINATION"

    return fingers, gesture


def face_actions(scores):
    checks = [
        ("BOTH_EYES_CLOSED",
         min(scores.get("eyeBlinkLeft", 0),
             scores.get("eyeBlinkRight", 0)), 0.55),
        ("LEFT_EYE_BLINK", scores.get("eyeBlinkLeft", 0), 0.55),
        ("RIGHT_EYE_BLINK", scores.get("eyeBlinkRight", 0), 0.55),
        ("MOUTH_OPEN", scores.get("jawOpen", 0), 0.40),
        ("SMILE", min(scores.get("mouthSmileLeft", 0),
                      scores.get("mouthSmileRight", 0)), 0.35),
        ("BROWS_RAISED", scores.get("browInnerUp", 0), 0.40),
        ("BROWS_DOWN",
         max(scores.get("browDownLeft", 0),
             scores.get("browDownRight", 0)), 0.45),
        ("LIPS_PUCKER", scores.get("mouthPucker", 0), 0.40),
        ("CHEEK_PUFF", scores.get("cheekPuff", 0), 0.40),
        ("EYES_WIDE",
         max(scores.get("eyeWideLeft", 0),
             scores.get("eyeWideRight", 0)), 0.40),
        ("NOSE_SNEER",
         max(scores.get("noseSneerLeft", 0),
             scores.get("noseSneerRight", 0)), 0.40)
    ]

    return [
        {"name": name, "score": round(float(score), 3)}
        for name, score, threshold in checks
        if score >= threshold
    ]


def head_angles(matrix):
    if matrix is None:
        return None

    r = matrix[:3, :3]
    sy = math.hypot(float(r[0, 0]), float(r[1, 0]))

    pitch = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.atan2(-float(r[2, 0]), sy)
    roll = math.atan2(float(r[1, 0]), float(r[0, 0]))

    return {
        "pitch_deg": round(math.degrees(pitch), 1),
        "yaw_deg": round(math.degrees(yaw), 1),
        "roll_deg": round(math.degrees(roll), 1)
    }


def draw_face(frame, landmarks):
    h, w = frame.shape[:2]

    for p in landmarks:
        cv2.circle(frame, pixel(p, w, h), 1, (0, 255, 255), -1)

    for chain in FACE_LINES:
        for a, b in zip(chain, chain[1:]):
            cv2.line(
                frame,
                pixel(landmarks[a], w, h),
                pixel(landmarks[b], w, h),
                (255, 180, 0), 1
            )


def draw_hand(frame, landmarks, label, fingers):
    h, w = frame.shape[:2]

    for a, b in HAND_EDGES:
        cv2.line(
            frame,
            pixel(landmarks[a], w, h),
            pixel(landmarks[b], w, h),
            (255, 0, 255), 2
        )

    for i, p in enumerate(landmarks):
        cv2.circle(frame, pixel(p, w, h), 3, (0, 255, 0), -1)

    wrist = pixel(landmarks[0], w, h)
    cv2.putText(
        frame, label, wrist, cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )


def load_yolo_pose_model():
    """Uu tien nap yolov8n-pose.onnx (dynamic=True, rect=True -> ~44ms/frame = 22+ FPS)."""
    onnx_candidate = BASE / "yolov8n-pose.onnx"
    pt_candidate = BASE / "yolov8n-pose.pt"
    custom_path = os.getenv("YOLO_POSE_MODEL", "").strip()

    if custom_path and os.path.exists(custom_path):
        print(f"[VISION] Nap model YOLO tuy chinh: {custom_path}")
        return YOLO(custom_path, task="pose"), "yolo-onnx" if custom_path.endswith(".onnx") else "yolo"

    if onnx_candidate.exists():
        print(f"[VISION] Su dung ONNX Runtime sieu toc (Dynamic Rect): {onnx_candidate}")
        return YOLO(str(onnx_candidate), task="pose"), "yolo-onnx"

    pt_path = str(pt_candidate) if pt_candidate.exists() else "yolov8n-pose.pt"
    print(f"[VISION] Nap model PyTorch: {pt_path}")
    pt_model = YOLO(pt_path)
    try:
        import onnxruntime  # noqa: F401
        print("[VISION] Dang xuat ONNX dynamic=True de tang toc CPU gap 3 lan...")
        exported = pt_model.export(format="onnx", dynamic=True, simplify=False)
        if exported and os.path.exists(exported):
            return YOLO(exported, task="pose"), "yolo-onnx"
    except Exception as exc:
        print("[VISION] Khong xuat duoc ONNX, dung PyTorch:", exc)
    return pt_model, "yolo"


def camera_worker():
    global latest_jpg, state

    cap = None
    tracker = MultiPersonFallTracker()
    seizure_detector = SeizureMotionDetector()
    mqtt_publisher = VisionMqttPublisher()
    sos_detector = HandSOSDetector()
    frame_count = 0
    cached_face = None
    cached_hands = None
    cached_poses = []
    cached_worlds = []
    last_event = None
    fps_window = []

    pose_backend = os.getenv("POSE_BACKEND", "yolo" if HAVE_YOLO else "mediapipe").lower()
    yolo_model = None
    if pose_backend == "yolo" and HAVE_YOLO:
        yolo_model, pose_backend = load_yolo_pose_model()

    try:
        source_value = os.getenv("VIDEO_SOURCE", "0")
        source = int(source_value) if source_value.isdigit() else source_value

        cap = FastVideoReader(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video: {source}")

        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        mode = vision.RunningMode.VIDEO

        MAX_POSES = max(1, int(os.getenv("MAX_POSES", "6")))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base(
                model_asset_path=str(BASE / "pose_landmarker.task")
            ),
            running_mode=mode,
            num_poses=MAX_POSES,
            min_pose_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )

        hand_options = vision.HandLandmarkerOptions(
            base_options=base(
                model_asset_path=str(BASE / "hand_landmarker.task")
            ),
            running_mode=mode,
            num_hands=2
        )

        face_options = vision.FaceLandmarkerOptions(
            base_options=base(
                model_asset_path=str(BASE / "face_landmarker.task")
            ),
            running_mode=mode,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        yolo_imgsz = int(os.getenv("YOLO_IMGSZ", "800"))
        yolo_conf = float(os.getenv("YOLO_CONF", "0.14"))

        with (
            vision.PoseLandmarker.create_from_options(pose_options) as pose_ai,
            vision.HandLandmarker.create_from_options(hand_options) as hand_ai,
            vision.FaceLandmarker.create_from_options(face_options) as face_ai
        ):
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    with lock:
                        state["status"] = "VIDEO_ENDED_OR_CAMERA_ERROR"
                    break

                if isinstance(source, int):
                    frame = cv2.flip(frame, 1)

                h_orig, w_orig = frame.shape[:2]
                if max(h_orig, w_orig) > MAX_FRAME_DIM:
                    scale = float(MAX_FRAME_DIM) / max(h_orig, w_orig)
                    frame = cv2.resize(
                        frame,
                        (int(w_orig * scale), int(h_orig * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )

                frame_count += 1
                now = time.monotonic()
                timestamp = int(now * 1000)
                h, w = frame.shape[:2]

                # Chay YOLO ONNX rect=True (~44ms) moi AI_EVERY frame, giu nguyen tracking muot o frame xen ke
                run_pose_now = (frame_count % AI_EVERY == 1) or (AI_EVERY == 1) or (not cached_poses)
                if run_pose_now:
                    detected_poses = []
                    detected_worlds = []
                    if yolo_model is not None:
                        yolo_res = yolo_model(
                            frame,
                            imgsz=yolo_imgsz,
                            conf=yolo_conf,
                            rect=True,
                            verbose=False
                        )
                        if yolo_res and len(yolo_res) > 0 and yolo_res[0].keypoints is not None:
                            kpts = yolo_res[0].keypoints
                            for k_idx in range(len(kpts)):
                                xy = kpts.xy[k_idx].tolist()
                                conf = kpts.conf[k_idx].tolist() if kpts.conf is not None else None
                                mp_lm = yolo_to_mp_landmarks(xy, conf, w, h)
                                detected_poses.append(mp_lm)
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        pose_result = pose_ai.detect_for_video(image, timestamp)
                        detected_poses = pose_result.pose_landmarks or []
                        detected_worlds = pose_result.pose_world_landmarks or []
                    cached_poses = detected_poses
                    cached_worlds = detected_worlds
                else:
                    detected_poses = cached_poses
                    detected_worlds = cached_worlds

                hands_fresh = ENABLE_HAND and (frame_count % HAND_EVERY == 0)
                face_fresh = ENABLE_FACE and (frame_count % FACE_EVERY == 0)

                if hands_fresh or face_fresh:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if hands_fresh:
                        cached_hands = hand_ai.detect_for_video(image, timestamp)
                    if face_fresh:
                        cached_face = face_ai.detect_for_video(image, timestamp)

                body_data = []
                hand_data = []
                face_data = None
                fall_state = "NO_PERSON"
                fall_metrics = {}
                people_data = []

                if detected_poses:
                    tracker_results, global_fall, any_fall_triggered, triggered_info = (
                        tracker.update(detected_poses, w, h, now)
                    )
                    fall_state = global_fall

                    for res in tracker_results:
                        idx = res["pose_index"]
                        p = detected_poses[idx]
                        world = detected_worlds[idx] if idx < len(detected_worlds) else None
                        p_num = res["person_id"]
                        col = (0, 0, 255) if res["state"] == "SUSPECTED_FALL" else PERSON_COLORS[(p_num - 1) % len(PERSON_COLORS)]
                        label = f"#{p_num} {res['state']}"
                        draw_pose(frame, p, label=label, color=col)

                        serialized = serialize(p, world)
                        people_data.append({
                            "person_id": p_num,
                            "state": res["state"],
                            "metrics": res["metrics"],
                            "landmarks": serialized
                        })

                    if tracker_results:
                        body_data = people_data[0]["landmarks"]
                        fall_metrics = tracker_results[0]["metrics"]

                    if any_fall_triggered and triggered_info:
                        fall_metrics = triggered_info.get("metrics", {})
                        p_id = triggered_info.get("person_id", 1)
                        conf_val = float(fall_metrics.get("confidence", 0.85))
                        angle_val = fall_metrics.get("angle_deg", 0.0)
                        speed_val = fall_metrics.get("hip_speed", 0.0)
                        event_text_vi = (
                            f"Phat hien nguoi #{p_id} nghi te nga "
                            f"(Do tin cay: {conf_val * 100:.0f}%, goc nghieng: {angle_val} deg, toc do roi: {speed_val}/s)"
                        )
                        last_event = {
                            "id": str(uuid.uuid4()),
                            "type": "SUSPECTED_FALL",
                            "person_id": p_id,
                            "time": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "confidence": conf_val,
                            "algorithm": "multi_person_heuristic_v2",
                            "metrics": fall_metrics,
                            "event_text_vi": event_text_vi
                        }
                else:
                    tracker.update([], w, h, now)
                    fall_state = "NO_PERSON"

                seizure_pose = detected_poses[0] if detected_poses else None
                seizure_status, seizure_event = seizure_detector.update(
                    seizure_pose,
                    now
                )
                if seizure_event is not None:
                    print('[VISION] SUSPECTED CONVULSIVE MOTION (DEMO):',
                          seizure_event['event_id'])
                    mqtt_publisher.emit_event(seizure_event)

                if ENABLE_HAND and cached_hands and cached_hands.hand_landmarks:
                    for i, hand in enumerate(cached_hands.hand_landmarks):
                        label = cached_hands.handedness[i][0].category_name
                        world = cached_hands.hand_world_landmarks[i]

                        fingers, gesture = fingers_of(hand, w, h)
                        draw_hand(frame, hand, label, fingers)

                        hand_data.append({
                            "hand": label,
                            "gesture": gesture,
                            "fingers": fingers,
                            "landmarks": serialize(hand, world)
                        })

                if hands_fresh:
                    sos_status, sos_event = sos_detector.update(
                        hand_data, now
                    )
                else:
                    sos_status = sos_detector.status(now)
                    sos_event = None
                if sos_event is not None:
                    print('HAND SOS TEST EVENT:', sos_event)

                if ENABLE_FACE and cached_face and cached_face.face_landmarks:
                    face = cached_face.face_landmarks[0]
                    draw_face(frame, face)

                    categories = (
                        cached_face.face_blendshapes[0]
                        if cached_face.face_blendshapes
                        else []
                    )
                    scores = {
                        c.category_name: round(float(c.score), 4)
                        for c in categories
                    }

                    matrixes = cached_face.facial_transformation_matrixes
                    head = head_angles(matrixes[0]) if len(matrixes) else None

                    face_data = {
                        "landmarks": serialize(face),
                        "blendshapes": scores,
                        "actions": face_actions(scores),
                        "head": head
                    }

                now_done = time.monotonic()
                fps_window.append(now_done)
                while len(fps_window) > 1 and (now_done - fps_window[0]) > 1.2:
                    fps_window.pop(0)
                fps = round((len(fps_window) - 1) / max(now_done - fps_window[0], 1e-3), 1) if len(fps_window) > 1 else 0.0

                cv2.putText(
                    frame,
                    f"NGUOI:{len(people_data)} TAY:{len(hand_data)} "
                    f"ENGINE:{pose_backend.upper()}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )

                cv2.putText(
                    frame, f"FPS:{fps} FALL:{fall_state}",
                    (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255) if fall_state == "SUSPECTED_FALL"
                    else (0, 255, 0),
                    2
                )

                encoded, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 72]
                )

                mqtt_publisher.emit_event(last_event)
                mqtt_publisher.emit_event(sos_detector.last_event)

                with lock:
                    if encoded:
                        latest_jpg = jpg.tobytes()

                    state = {
                        "status": "RUNNING",
                        "fps": fps,
                        "people_count": len(people_data),
                        "people": people_data,
                        "body": body_data,
                        "hands": hand_data,
                        "face": face_data,
                        "fall_state": fall_state,
                        "fall_metrics": fall_metrics,
                        "sos": sos_status,
                        "last_sos_event": sos_detector.last_event,
                        "seizure": seizure_status,
                        "last_seizure_event": seizure_detector.last_event,
                        "last_event": last_event,
                        "mqtt": mqtt_publisher.status()
                    }
                jpg_event.set()

                mqtt_publisher.emit_status(state)

    except Exception as exc:
        print("FULL AI ERROR:", repr(exc))
        with lock:
            state["status"] = f"ERROR: {exc}"

    finally:
        if cap is not None:
            cap.release()
        mqtt_publisher.close()


@asynccontextmanager
async def lifespan(app):
    stop.clear()
    worker = threading.Thread(target=camera_worker, daemon=True)
    worker.start()

    yield

    stop.set()
    worker.join(timeout=3)


app = FastAPI(
    title="MediAssistRover Full Human Tracking",
    lifespan=lifespan
)


@app.get("/")
def home():
    return FileResponse(BASE / "full_ai.html")


@app.get("/api/state")
def api_state():
    with lock:
        return dict(state)


def stream_frames():
    while not stop.is_set():
        jpg_event.wait(timeout=0.05)
        jpg_event.clear()
        with lock:
            jpg = latest_jpg

        if jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg + b"\r\n"
            )


@app.get("/video")
def video():
    return StreamingResponse(
        stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
