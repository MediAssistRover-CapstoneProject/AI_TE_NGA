"""Bộ phát hiện nghi té ngã đa giai đoạn (Multi-phase Fall Detector v2).

Đặc điểm cải tiến:
1. Phân biệt té ngã với các hoạt động bình thường:
   - Cúi người (Bending): Thân nghiêng nhưng hông ở vị trí cao, không có gia tốc rơi.
   - Ngồi ghế (Sitting): Hông hạ xuống nhưng thân vẫn thẳng đứng (< 35 độ).
   - Nằm từ từ (Lying down slowly): Thân nằm ngang nhưng không có gia tốc rơi nhanh.
   - Té ngã thật (Fall): Rơi nhanh đột ngột (hip_speed > threshold) -> thân lật ngang -> nằm bất động ở sàn.
2. Tính toán điểm tin cậy (confidence score: 0.50 - 0.98), phục vụ luồng WF_FALL_DETECTION_RESPONSE.
3. Cấu hình linh hoạt qua biến môi trường (FALL_ANGLE_THRESHOLD, FALL_SPEED_THRESHOLD...).
"""
import math
import os


HIP_JOINTS = (23, 24)
TORSO_JOINTS = (11, 12) + HIP_JOINTS
BODY_JOINTS = TORSO_JOINTS + (25, 26, 27, 28)


def visible_image_point(point):
    """Return an observable keypoint inside the normalized image."""
    if isinstance(point, dict):
        x, y = point.get("x"), point.get("y")
        visibility = point.get("visibility", 1.0)
    else:
        x, y = getattr(point, "x", None), getattr(point, "y", None)
        visibility = getattr(point, "visibility", 1.0)
    try:
        x, y, visibility = float(x), float(y), float(visibility)
    except (TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(value) for value in (x, y, visibility))
            or not (0. <= x < 1. and 0. <= y < 1.) or visibility < .5):
        return None
    return x, y


def observed_body_axis(points, width, height):
    """Use the torso when visible, otherwise a visible extended lower leg."""
    visible = {i: visible_image_point(points[i]) for i in BODY_JOINTS if i < len(points)}
    if any(visible.get(i) is None for i in HIP_JOINTS):
        return None
    hips = tuple((visible[23][axis] + visible[24][axis]) / 2 for axis in (0, 1))
    if all(visible.get(i) is not None for i in (11, 12)):
        shoulders = tuple((visible[11][axis] + visible[12][axis]) / 2 for axis in (0, 1))
        return visible, hips, shoulders, 'TORSO'
    legs = []
    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        if any(visible.get(i) is None for i in (hip, knee, ankle)):
            continue
        a, b, c = (visible[i] for i in (hip, knee, ankle))
        upper = ((a[0] - b[0]) * width, (a[1] - b[1]) * height)
        lower = ((c[0] - b[0]) * width, (c[1] - b[1]) * height)
        lengths = math.hypot(*upper), math.hypot(*lower)
        if min(lengths) < 10:
            continue
        cosine = sum(u * v for u, v in zip(upper, lower)) / (lengths[0] * lengths[1])
        knee_angle = math.degrees(math.acos(max(-1., min(1., cosine))))
        if knee_angle >= 145.:
            legs.append((a, c))
    if not legs:
        return None
    hip, ankle = max(legs, key=lambda p: math.hypot((p[0][0]-p[1][0])*width, (p[0][1]-p[1][1])*height))
    return visible, hips, (hips[0] + ankle[0] - hip[0], hips[1] + ankle[1] - hip[1]), 'LOWER_BODY'


class FallDetector:
    def __init__(self):
        self.angle_threshold = float(os.getenv("FALL_ANGLE_THRESHOLD", "50.0"))
        self.speed_threshold = float(os.getenv("FALL_SPEED_THRESHOLD", "0.20"))
        self.confirm_seconds = float(os.getenv("FALL_CONFIRM_SEC", "0.5"))
        self.cooldown_seconds = float(os.getenv("FALL_COOLDOWN_SEC", "20.0"))
        self.alert_duration = float(os.getenv("FALL_ALERT_DURATION", "7.0"))
        self.floor_enabled = os.getenv("FLOOR_POSTURE_ENABLED", "true").lower() == "true"
        self.floor_confirm_seconds = float(os.getenv("FLOOR_POSTURE_CONFIRM_SEC", "2.0"))
        self.floor_angle_threshold = float(os.getenv("FLOOR_POSTURE_ANGLE_DEG", "70.0"))
        self.floor_min_hip_y = float(os.getenv("FLOOR_POSTURE_MIN_HIP_Y", "0.45"))
        if (not math.isfinite(self.floor_confirm_seconds) or not .5 <= self.floor_confirm_seconds <= 30.
                or not math.isfinite(self.floor_angle_threshold) or not 60. <= self.floor_angle_threshold <= 90.
                or not math.isfinite(self.floor_min_hip_y) or not 0. <= self.floor_min_hip_y < 1.):
            raise ValueError("Invalid FLOOR_POSTURE configuration")

        self.prev = None
        self.motion_until = 0.0
        self.peak_speed = 0.0
        self.candidate_since = None
        self.cooldown_until = 0.0
        self.alert_until = 0.0
        self.last_confidence = 0.0
        self.floor_since = None
        self.floor_model = None
        self.floor_reported = False
        self.last_detection_kind = None

    def reset_pose(self):
        self.prev = None
        self.candidate_since = None
        self.motion_until = 0.0
        self.peak_speed = 0.0
        self.floor_since = None
        self.floor_model = None
        self.floor_reported = False

    def update(self, points, width, height, now, allow_transition=True):
        body = observed_body_axis(points, width, height)
        if body is None:
            self.reset_pose()
            hips = [visible_image_point(points[i]) if i < len(points) else None for i in HIP_JOINTS]
            if all(point is not None for point in hips):
                # Preserve the starting hip position when a close camera crops
                # the shoulders/feet. This alone cannot confirm a horizontal body.
                self.prev = (now, (hips[0][1] + hips[1][1]) / 2, None, 'HIP_ONLY')
            return "LOW_VISIBILITY", {}, False
        visible, (hip_x, hip_y), (axis_x, axis_y), body_model = body

        # Góc thân người (torso angle) so với phương thẳng đứng (0 = đứng thẳng, 90 = nằm ngang)
        dx = (axis_x - hip_x) * width
        dy = (axis_y - hip_y) * height
        angle = math.degrees(math.atan2(abs(dx), max(abs(dy), 0.001)))

        # Aspect ratio của bounding box cơ thể (rộng / cao)
        # Đứng: width < height (ratio < 0.65). Ngã ngang: ratio >= 0.95
        xs, ys = zip(*(point for point in visible.values() if point is not None))
        box_w = (max(xs) - min(xs)) * width
        box_h = (max(ys) - min(ys)) * height
        aspect_ratio = round(box_w / max(box_h, 1.0), 2)

        hip_speed = 0.0
        angular_speed = 0.0

        if self.prev is not None:
            old_t, old_hip_y, old_angle, old_model = self.prev
            dt = now - old_t
            if dt <= 0. or dt > 1. or old_model != body_model:
                self.floor_since, self.floor_model = None, None

            if .015 < dt <= 1.:
                # Vận tốc rơi theo trục Y (dương = rơi xuống sàn)
                hip_speed = (hip_y - old_hip_y) / dt
                angular_speed = abs(angle - old_angle) / dt if old_model == body_model else 0.
                if old_model != body_model:
                    self.candidate_since = None

                # Phát hiện rơi nhanh hoặc xoay lật đột ngột
                rapid_drop = hip_speed > self.speed_threshold
                rapid_rotation = angular_speed > 85.0 and hip_speed > 0.05

                if allow_transition and (rapid_drop or rapid_rotation):
                    self.motion_until = now + 1.8
                    self.peak_speed = max(self.peak_speed, hip_speed)

        if not allow_transition:
            # Image displacement is not patient motion while the camera moves.
            self.motion_until = 0.0
            self.candidate_since = None
            self.peak_speed = 0.0

        self.prev = (now, hip_y, angle, body_model)

        # Tiêu chí tư thế ngang: góc >= ngưỡng HOẶC tỷ lệ khung bao ngang >= 0.95
        is_horizontal = (angle >= self.angle_threshold) or (aspect_ratio >= 0.95 and angle >= 40.0)
        if body_model == 'LOWER_BODY':
            is_horizontal = angle >= max(65., self.angle_threshold)

        # Tiêu chí độ cao thấp: hông ở nửa dưới khung hình (gần mặt sàn)
        # Loại trừ trường hợp cúi nhặt đồ nhưng hông vẫn ở vị trí đứng cao
        is_low_elevation = hip_y > 0.45

        suspected = False

        # Chỉ kích hoạt nếu có giai đoạn rơi nhanh trước đó VÀ hiện tại đang nằm ngang gần sàn
        if is_horizontal and is_low_elevation and now < self.motion_until:
            if self.candidate_since is None:
                self.candidate_since = now

            immobility_duration = now - self.candidate_since

            # Xác nhận tư thế nằm bất động sau va chạm trong khoảng confirm_seconds
            if immobility_duration >= self.confirm_seconds and now >= self.cooldown_until:
                suspected = True
                self.last_detection_kind = "FALL_TRANSITION"
                self.floor_reported = True
                self.cooldown_until = now + self.cooldown_seconds
                self.alert_until = now + self.alert_duration
                self.candidate_since = None

                # Điểm tin cậy (Confidence Score 0.55 - 0.98)
                score_angle = 0.35 * min(1.0, max(0.0, angle / 75.0))
                score_speed = 0.35 * min(1.0, max(0.0, self.peak_speed / 0.40))
                score_aspect = 0.20 * min(1.0, max(0.0, aspect_ratio / 1.1))
                score_duration = 0.10 * min(1.0, immobility_duration / 1.0)
                self.last_confidence = round(
                    min(0.98, max(0.55, score_angle + score_speed + score_aspect + score_duration)), 2
                )
                self.peak_speed = 0.0
        else:
            # Confirmation requires a continuous low, horizontal posture after motion.
            self.candidate_since = None
            if not is_horizontal:
                self.peak_speed = 0.0

        # RGB supplies a suspicion only. The caller must verify this person's
        # hip height using synchronized registered depth before reporting it.
        floor_posture = (self.floor_enabled and angle >= self.floor_angle_threshold
                         and hip_y > self.floor_min_hip_y)
        if floor_posture:
            if self.floor_since is None or self.floor_model != body_model:
                self.floor_since, self.floor_model = now, body_model
            floor_ready = now - self.floor_since >= self.floor_confirm_seconds
            if floor_ready and not self.floor_reported and now >= self.cooldown_until:
                suspected = True
                self.floor_reported = True
                self.last_detection_kind = "PERSON_ON_FLOOR"
                self.last_confidence = round(min(.80, .55 + .20 * angle / 90.), 2)
                self.cooldown_until = now + self.cooldown_seconds
        else:
            self.floor_since, self.floor_model, self.floor_reported = None, None, False
            floor_ready = False

        if self.last_detection_kind == "FALL_TRANSITION" and now < self.alert_until:
            state = "SUSPECTED_FALL"
        elif floor_ready and self.last_detection_kind == "PERSON_ON_FLOOR":
            state = "SUSPECTED_PERSON_ON_FLOOR"
        elif self.candidate_since is not None:
            state = "CHECKING"
        elif is_horizontal and is_low_elevation:
            state = "HORIZONTAL_POSTURE"
        elif angle >= 40.0:
            state = "BENDING_OR_LEANING"
        else:
            state = "POSE_DETECTED"

        metrics = {
            "torso_angle_deg": round(angle, 2) if body_model == 'TORSO' else None,
            "body_axis_angle_deg": round(angle, 2),
            "body_model": body_model,
            "hip_y_normalized": round(hip_y, 4),
            "hip_speed_normalized_per_sec": round(hip_speed, 3),
            "angular_speed_deg_per_sec": round(angular_speed, 2),
            "aspect_ratio": aspect_ratio,
            "confidence": self.last_confidence if state in ("SUSPECTED_FALL", "SUSPECTED_PERSON_ON_FLOOR") else None,
            "detection_kind": self.last_detection_kind if state in ("SUSPECTED_FALL", "SUSPECTED_PERSON_ON_FLOOR") else None,
            "ground_confirmation_required": state == "SUSPECTED_PERSON_ON_FLOOR",
            "phase": state,
        }

        return state, metrics, suspected


class MultiPersonFallTracker:
    """Theo dõi và phát hiện té ngã độc lập cho từng người trong khung hình (Multi-person)."""

    def __init__(self, max_distance=0.35, timeout_sec=2.0):
        self.max_dist = max_distance
        self.timeout = timeout_sec
        self.tracks = {}
        self.next_person_id = 1

    def update(self, all_poses, width, height, now, allow_transition=True):
        indexed_poses = [(index, pose) for index, pose in enumerate(all_poses) if all(
            i < len(pose) and visible_image_point(pose[i]) is not None for i in HIP_JOINTS)]
        original_indices = [index for index, _ in indexed_poses]
        all_poses = [pose for _, pose in indexed_poses]
        if not all_poses:
            to_del = [tid for tid, t in self.tracks.items() if now - t["seen"] > self.timeout]
            for tid in to_del:
                del self.tracks[tid]
            return [], "NO_PERSON", False, None

        centers = []
        for pose in all_poses:
            def _get(p):
                if isinstance(p, dict):
                    return float(p.get("x", 0.5)), float(p.get("y", 0.5))
                return float(p.x), float(p.y)
            p23_x, p23_y = _get(pose[23]) if len(pose) > 23 else (0.5, 0.5)
            p24_x, p24_y = _get(pose[24]) if len(pose) > 24 else (0.5, 0.5)
            centers.append(((p23_x + p24_x) / 2.0, (p23_y + p24_y) / 2.0))

        assigned = {}
        unassigned_poses = set(range(len(all_poses)))

        for tid, t in list(self.tracks.items()):
            best_idx = None
            best_d = self.max_dist
            for i in unassigned_poses:
                d = math.hypot(centers[i][0] - t["pos"][0], centers[i][1] - t["pos"][1])
                if d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None:
                assigned[tid] = best_idx
                unassigned_poses.remove(best_idx)
                self.tracks[tid]["pos"] = centers[best_idx]
                self.tracks[tid]["seen"] = now
            elif now - t["seen"] > self.timeout:
                del self.tracks[tid]

        for i in unassigned_poses:
            tid = self.next_person_id
            self.next_person_id += 1
            self.tracks[tid] = {
                "detector": FallDetector(),
                "pos": centers[i],
                "seen": now,
                "person_id": tid
            }
            assigned[tid] = i

        results = []
        any_triggered = False
        triggered_info = None
        has_fall = False
        has_checking = False
        has_horizontal = False

        for tid, pose_idx in sorted(assigned.items(), key=lambda x: x[0]):
            detector = self.tracks[tid]["detector"]
            person_id = self.tracks[tid]["person_id"]
            pose = all_poses[pose_idx]
            center = centers[pose_idx]
            ambiguous = any(index != pose_idx and math.hypot(center[0]-other[0], center[1]-other[1]) < .12
                            for index, other in enumerate(centers))
            if ambiguous:
                detector.reset_pose()
                state, metrics, triggered = 'AMBIGUOUS_PERSON', {}, False
            else:
                state, metrics, triggered = detector.update(pose, width, height, now, allow_transition)

            if triggered:
                any_triggered = True
                triggered_info = {
                    "person_id": person_id,
                    "metrics": metrics,
                    "center": centers[pose_idx]
                }

            if state == "SUSPECTED_FALL":
                has_fall = True
            elif state == "CHECKING":
                has_checking = True
            elif state == "HORIZONTAL_POSTURE":
                has_horizontal = True

            results.append({
                "person_id": person_id,
                "pose_index": original_indices[pose_idx],
                "state": state,
                "metrics": metrics,
                "triggered": triggered,
                "center": centers[pose_idx]
            })

        if has_fall:
            global_state = "SUSPECTED_FALL"
        elif has_checking:
            global_state = "CHECKING"
        elif has_horizontal:
            global_state = "HORIZONTAL_POSTURE"
        elif results:
            global_state = f"{len(results)}_PEOPLE_DETECTED"
        else:
            global_state = "NO_PERSON"

        return results, global_state, any_triggered, triggered_info
