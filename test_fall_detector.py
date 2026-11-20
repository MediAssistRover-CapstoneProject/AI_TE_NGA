"""Unit tests for FallDetector v2.

Tests both positive controls (actual simulated falls) and negative controls
(sitting down, bending to pick up an item, slowly lying down, normal walking).
"""
import math
from fall_detector import FallDetector


def create_pose(hip_y, torso_angle_deg, visibility=0.99):
    """Generate 33 MediaPipe pose landmarks from hip_y and torso angle."""
    pts = [{"x": 0.5, "y": 0.5, "visibility": visibility} for _ in range(33)]

    # Shoulder distance from hip in normalized coordinates (approx 0.25)
    torso_len = 0.25
    rad = math.radians(torso_angle_deg)
    dx = torso_len * math.sin(rad)
    dy = torso_len * math.cos(rad)

    shoulder_y = hip_y - dy
    shoulder_x = 0.5 + dx

    # Left and right shoulders (11, 12)
    pts[11].update({"x": shoulder_x - 0.08, "y": shoulder_y})
    pts[12].update({"x": shoulder_x + 0.08, "y": shoulder_y})

    # Left and right hips (23, 24)
    pts[23].update({"x": 0.5 - 0.06, "y": hip_y})
    pts[24].update({"x": 0.5 + 0.06, "y": hip_y})

    # Knees (25, 26) and Ankles (27, 28)
    leg_len = 0.35
    pts[25].update({"x": 0.48, "y": min(0.98, hip_y + leg_len * 0.5)})
    pts[26].update({"x": 0.52, "y": min(0.98, hip_y + leg_len * 0.5)})
    pts[27].update({"x": 0.48, "y": min(0.99, hip_y + leg_len)})
    pts[28].update({"x": 0.52, "y": min(0.99, hip_y + leg_len)})

    return pts


def simulate_scenario(generator_fn, duration_sec=5.0, fps=20):
    detector = FallDetector()
    events = []
    states = []

    steps = int(duration_sec * fps)
    for i in range(steps):
        t = i / fps
        points, w, h = generator_fn(t)
        state, metrics, triggered = detector.update(points, w, h, t)
        states.append(state)
        if triggered:
            events.append({"time": t, "metrics": metrics})

    return events, states


def test_positive_fall():
    """Scenario 1: Actual fall.
    - t=0..1.0s: Standing upright (angle=5°, hip_y=0.45)
    - t=1.0..1.4s: Rapid descent & rotation (hip_y 0.45 -> 0.78, angle 5° -> 78°)
    - t=1.4..3.5s: Lying on floor (hip_y=0.78, angle=80°)
    """
    def sim(t):
        if t < 1.0:
            hip_y = 0.45
            angle = 5.0
        elif t < 1.4:
            prog = (t - 1.0) / 0.4
            hip_y = 0.45 + 0.33 * prog
            angle = 5.0 + 73.0 * prog
        else:
            hip_y = 0.78
            angle = 80.0
        return create_pose(hip_y, angle), 640, 480

    events, states = simulate_scenario(sim, duration_sec=3.5)
    assert len(events) == 1, f"Expected exactly 1 fall event, got {len(events)}"
    conf = events[0]["metrics"]["confidence"]
    assert conf is not None and conf >= 0.75, f"Expected confidence >= 0.75, got {conf}"
    print(f"  [PASS] Positive fall detected! Confidence: {conf}")


def test_negative_bending():
    """Scenario 2: Bending down to tie shoes / pick up item.
    - Torso bends forward (angle reaches 65°), but hip remains elevated (hip_y <= 0.42).
    - Must NOT trigger fall.
    """
    def sim(t):
        # Bending down and coming back up
        angle = 10.0 + 55.0 * math.sin(min(math.pi, t * 1.5))
        hip_y = 0.40  # Hips stay high
        return create_pose(hip_y, angle), 640, 480

    events, states = simulate_scenario(sim, duration_sec=3.0)
    assert len(events) == 0, f"False positive on bending: {len(events)} events"
    print("  [PASS] Negative control: Bending down correctly rejected")


def test_negative_sitting():
    """Scenario 3: Sitting down in a chair.
    - Hips descend (hip_y 0.42 -> 0.65), but torso stays upright (angle < 25°).
    - Must NOT trigger fall.
    """
    def sim(t):
        prog = min(1.0, t / 0.6)
        hip_y = 0.42 + 0.23 * prog
        angle = 8.0 + 10.0 * math.sin(prog * math.pi)  # Slight upright lean
        return create_pose(hip_y, angle), 640, 480

    events, states = simulate_scenario(sim, duration_sec=3.0)
    assert len(events) == 0, f"False positive on sitting down: {len(events)} events"
    print("  [PASS] Negative control: Sitting down correctly rejected")


def test_negative_slow_lying():
    """Scenario 4: Slowly lying down in bed.
    - Torso smoothly transitions to 75° over 3 seconds (speed < 0.10/s).
    - No rapid impact phase -> Must NOT trigger fall.
    """
    def sim(t):
        prog = min(1.0, t / 3.0)
        hip_y = 0.45 + 0.30 * prog
        angle = 5.0 + 75.0 * prog
        return create_pose(hip_y, angle), 640, 480

    events, states = simulate_scenario(sim, duration_sec=4.0)
    assert len(events) == 0, f"False positive on slow lying down: {len(events)} events"
    print("  [PASS] Negative control: Slowly lying down correctly rejected")


def test_already_lying_requires_ground_confirmation():
    """Already horizontal produces a separate suspicion after sustained view."""
    events, _ = simulate_scenario(lambda t: (create_pose(.75, 80.), 640, 480), duration_sec=3.)
    assert len(events) == 1
    event = events[0]
    assert event["time"] >= 2.
    assert event["metrics"]["detection_kind"] == "PERSON_ON_FLOOR"
    assert event["metrics"]["ground_confirmation_required"] is True
    print("  [PASS] Already lying: sustained RGB suspicion requires depth floor check")


def test_floor_posture_brief_or_occluded_does_not_accumulate():
    """Brief/hidden poses cannot complete the sustained horizontal window."""
    detector = FallDetector()
    triggers = []
    for index in range(12):
        now = index * .25
        visible = index not in (4, 8)
        pose = create_pose(.75, 80., visibility=.99 if visible else .1)
        _, _, triggered = detector.update(pose, 640, 480, now)
        triggers.append(triggered)
    assert not any(triggers)
    print("  [PASS] Brief/occluded horizontal posture cannot complete confirmation")


def test_lower_body_already_lying_and_folded_legs():
    """A clear extended lower body works; hips alone or folded legs do not."""
    def lower_pose(folded=False, hips_only=False):
        points = [{"x": .5, "y": .5, "visibility": 0.} for _ in range(33)]
        for hip, knee, ankle, y in ((23, 25, 27, .70), (24, 26, 28, .76)):
            points[hip] = {"x": .3, "y": y, "visibility": .99}
            if not hips_only:
                points[knee] = {"x": .5, "y": y, "visibility": .99}
                points[ankle] = {"x": .5 if folded else .75,
                                 "y": y + .15 if folded else y, "visibility": .99}
        return points
    for kwargs, should_trigger in (({}, True), ({"folded": True}, False), ({"hips_only": True}, False)):
        detector = FallDetector()
        events = [detector.update(lower_pose(**kwargs), 640, 480, index * .25)[2] for index in range(13)]
        assert any(events) is should_trigger, (kwargs, events)
    print("  [PASS] Lower-body posture distinguishes clear legs from folded/missing legs")


def test_moving_camera_suppresses_transition_not_ground_candidate():
    detector = FallDetector()
    detected = []
    for index in range(20):
        now = index * .25
        hip, angle = ((.4, 5.) if index < 2 else (.75, 80.))
        _, metrics, triggered = detector.update(create_pose(hip, angle), 640, 480, now, allow_transition=False)
        if triggered:
            detected.append(metrics)
    assert detected and all(item["detection_kind"] == "PERSON_ON_FLOOR" for item in detected)
    assert all(item["ground_confirmation_required"] for item in detected)
    print("  [PASS] Moving camera cannot manufacture a FALL_TRANSITION event")


def test_overlay_uses_one_pose_inference_and_keeps_partial_people_boxes():
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from fall_service import PoseModel, overlay_result

    # Raw YOLO boxes are in the 416-square letterbox. The first person is
    # clipped by the image border and has no trustworthy hips.
    partial = np.zeros(56, dtype=np.float32)
    partial[:5] = [40., 160., 160., 240., .9]
    whole = np.zeros(56, dtype=np.float32)
    whole[:5] = [312., 208., 100., 200., .8]
    points = whole[5:].reshape(17, 3)
    points[:] = [312., 208., .95]
    points[11] = [300., 230., .95]
    points[12] = [330., 230., .95]
    rows = np.stack([partial, whole], axis=1)[None]
    calls = []
    def infer(names, tensors):
        calls.append(tensors)
        return [rows]
    model = PoseModel.__new__(PoseModel)
    model.input_name = 'images';model.session = SimpleNamespace(run=infer)
    with Image.new('RGB', (640, 480), (127, 127, 127)) as frame:
        poses = model.poses(frame)
    assert len(calls) == 1 and len(poses) == 1 and len(poses.detections) == 2
    assert poses.detections[0]['pose_index'] is None
    overlay = overlay_result(poses, [{'pose_index': 0, 'person_id': 7, 'state': 'CHECKING'}],
                             640, 480, 1000., False)
    partial_box, tracked = overlay['detections']
    assert partial_box['track_id'] is None and partial_box['state'] == 'PERSON'
    assert partial_box['bbox'][0] == 0. and partial_box['bbox'][1] == 0.
    assert tracked['track_id'] == 7 and tracked['state'] == 'OBSERVING'
    for detection in overlay['detections']:
        x, y, width, height = detection['bbox']
        assert 0. <= x < x + width <= 1. and 0. <= y < y + height <= 1.
        assert 0. <= detection['confidence'] <= 1.
    # A crowded image remains bounded without a second person model.
    rows = np.stack([np.array([20.+index*38., 208., 12., 20., .9] + [0.]*51,
                              dtype=np.float32) for index in range(10)], axis=1)[None]
    with Image.new('RGB', (640, 480)) as frame:
        assert len(model.poses(frame).detections) == 8
    assert len(calls) == 2
    print('  [PASS] Overlay retains partial people, clips boxes and uses one pose inference per frame')


def test_overlay_cache_keeps_capture_time_and_new_empty_frame_clears_boxes():
    from io import BytesIO
    from PIL import Image
    from fall_service import FallService, PoseBatch

    first = PoseBatch();first.extend([create_pose(.45, 5.)])
    first.detections = [{'pose_index': 0, 'bbox': [.2, .1, .4, .8], 'confidence': .9}]
    shifted = PoseBatch();shifted.extend([create_pose(.45, 5.)])
    shifted.detections = [{'pose_index': 0, 'bbox': [.25, .1, .4, .8], 'confidence': .9}]
    class Model:
        def __init__(self):self.calls = 0
        def poses(self, frame):
            self.calls += 1
            return [first, shifted, PoseBatch()][self.calls-1]
    model = Model();service = FallService(model);stream = BytesIO()
    with Image.new('RGB', (640, 480)) as frame:frame.save(stream, format='JPEG')
    jpeg = stream.getvalue()
    initial = service.observe('overlay', jpeg, 10., 1000.)
    cached = service.observe('overlay', jpeg, 10.1, 1000.)
    assert model.calls == 1 and cached['overlay'] == initial['overlay']
    assert cached['overlay']['captured_at'] == '1970-01-01T00:16:40+00:00'
    fresh = service.observe('overlay', jpeg, 10.2, 1000.2)
    assert model.calls == 2 and fresh['overlay']['captured_at'] != initial['overlay']['captured_at']
    assert fresh['overlay']['detections'][0]['bbox'] != initial['overlay']['detections'][0]['bbox']
    assert fresh['overlay']['detections'][0]['track_id'] == initial['overlay']['detections'][0]['track_id']
    empty = service.observe('overlay', jpeg, 10.4, 1000.4)
    assert model.calls == 3 and empty['overlay']['detections'] == []
    assert service.observe('overlay', jpeg, 10.5, 1000.4)['overlay']['detections'] == []
    assert model.calls == 3 and empty['suspected_fall'] is False
    print('  [PASS] Overlay cache preserves capture time; a fresh empty frame clears old boxes')


def test_overlay_floor_state_matches_existing_fall_suspicion_and_clears():
    from io import BytesIO
    from PIL import Image
    from fall_service import FallService, PoseBatch

    class Model:
        present = True
        def poses(self, frame):
            batch = PoseBatch()
            if self.present:
                batch.append(create_pose(.75, 80.))
                batch.detections = [{'pose_index': 0, 'bbox': [.1, .1, .8, .8], 'confidence': .9}]
            return batch
    stream = BytesIO()
    with Image.new('RGB', (640, 480)) as frame:frame.save(stream, format='JPEG')
    jpeg = stream.getvalue();model = Model();service = FallService(model)
    for index in range(13):
        value = service.observe('floor', jpeg, 10.+index*.25, 1000.+index*.25)
    assert value['suspected_fall'] is True and value['detection_kind'] == 'PERSON_ON_FLOOR'
    assert value['ground_confirmation_required'] is True
    assert value['visual_review_eligible'] is True
    assert value['overlay']['detections'][0]['state'] == 'PERSON_ON_FLOOR'
    assert len(value['person']['keypoints']) == 2
    model.present = False
    empty = service.observe('floor', jpeg, 13.25, 1003.25)
    assert empty['overlay']['detections'] == [] and empty['suspected_fall'] is False
    print('  [PASS] Overlay floor state matches the existing depth-confirmation-required suspicion contract')


def test_transition_does_not_remain_latched_without_the_person():
    from io import BytesIO
    from PIL import Image
    from fall_service import FallService, frame_context

    class Model:
        index = 0
        def poses(self, frame):
            t = self.index / 20.
            self.index += 1
            if t >= 2.:
                return []
            progress = min(1., max(0., (t - 1.) / .4))
            return [create_pose(.45 + .33 * progress, 5. + 75. * progress)]

    service = FallService(Model())
    with BytesIO() as stream, Image.new('RGB', (320, 240)) as frame:
        frame.save(stream, format='JPEG')
        jpeg = stream.getvalue()
    answers = [service.observe('transition', jpeg, 10.+index*.05, 1000.+index*.05) for index in range(42)]
    assert any(answer.get('detection_kind') == 'FALL_TRANSITION' for answer in answers[:40])
    assert answers[-1]['suspected_fall'] is False
    assert frame_context({'X-Fall-Captured-At': '1970-01-01T00:16:40Z'}, now=1003.3) == (1000., False)
    print('  [PASS] Stale transition clears when the person leaves, small clock skew keeps ordered frames')


def test_cropped_bent_leg_tracks_same_person_and_requires_depth():
    from io import BytesIO
    from PIL import Image
    from fall_service import FallService, PoseBatch

    class Model:
        index = 0

        def poses(self, frame):
            pose = [{"x": .5, "y": .5, "visibility": 0.} for _ in range(33)]
            landmarks = {23: (.3, .75), 25: (.5, .75), 27: (.47, .80)}
            if self.index == 0:
                landmarks[24] = (.32, .78)
            elif self.index >= 12:
                # A seated person's shin descends vertically from the knee.
                landmarks[27] = (.5, .96)
            for joint, (x, y) in landmarks.items():
                pose[joint] = {"x": x, "y": y, "visibility": .9}
            self.index += 1
            batch = PoseBatch()
            batch.append(pose)
            batch.detections = [{"pose_index": 0, "bbox": [.15, .65, .6, .32], "confidence": .9}]
            return batch

    service = FallService(Model())
    with BytesIO() as stream, Image.new('RGB', (640, 480)) as frame:
        frame.save(stream, format='JPEG')
        jpeg = stream.getvalue()
    values = [service.observe('cropped', jpeg, 10.+i*.25, 1000.+i*.25) for i in range(24)]
    assert all(value["overlay"]["detections"][0]["track_id"] == 1 for value in values)
    assert not any(value["suspected_fall"] for value in values[:8])
    assert values[8]["detection_kind"] == "PERSON_ON_FLOOR"
    assert values[8]["ground_confirmation_required"] is True
    assert values[8]["visual_review_eligible"] is False
    assert values[8]["person"]["keypoints"] == [{"x": .3, "y": .75}, {"x": .5, "y": .75}]
    assert not any(value["suspected_fall"] for value in values[12:])
    print('  [PASS] Cropped bent-leg track stays identified, requires depth, and rejects a seated shin')


def test_static_floor_timer_survives_a_change_of_observable_body_model():
    detector = FallDetector(allow_partial=True)
    events = []
    for i in range(12):
        pose = create_pose(.75, 80.)
        if i % 2:
            pose[12]['visibility'] = 0.
        state, metrics, triggered = detector.update(pose, 640, 480, i*.25)
        if triggered:
            events.append((i*.25, metrics))
    assert len(events) == 1 and events[0][0] == 2.
    assert events[0][1]['detection_kind'] == 'PERSON_ON_FLOOR'


def test_end_on_body_is_a_depth_candidate_only_and_never_an_rgb_fall():
    from io import BytesIO
    from PIL import Image
    from fall_service import FallService, PoseBatch

    class Model:
        missing = False
        def poses(self, frame):
            batch = PoseBatch()
            if self.missing:
                return batch
            pose = [{"x": .5, "y": .5, "visibility": 0.} for _ in range(33)]
            # Closed legs point along the camera axis: vertically aligned in RGB.
            for joint, point in {23:(.47,.55),24:(.51,.55),25:(.48,.72),27:(.49,.92)}.items():
                pose[joint] = dict(x=point[0],y=point[1],visibility=.95)
            batch.append(pose)
            batch.detections = [dict(pose_index=0,bbox=[.4,.5,.2,.49],confidence=.9)]
            return batch

    model = Model();service = FallService(model)
    with BytesIO() as stream, Image.new('RGB',(640,480)) as frame:
        frame.save(stream,format='JPEG');jpeg=stream.getvalue()
    results = [service.observe('end-on',jpeg,10+i*.25,1000+i*.25) for i in range(10)]
    assert not any(r['suspected_fall'] for r in results)
    assert not any('floor_candidate' in r for r in results[:8])
    candidate = results[8]['floor_candidate']
    assert candidate['person']['keypoint_roles'] == ['hip','ankle']
    assert candidate['ground_confirmation_required'] and not candidate['visual_review_eligible']
    model.missing=True
    assert 'floor_candidate' not in service.observe('end-on',jpeg,13.,1003.)
    model.missing=False
    assert 'floor_candidate' not in service.observe('end-on',jpeg,13.25,1003.25)
    moving = [service.observe('moving-end-on',jpeg,20+i*.25,2000+i*.25,True) for i in range(10)]
    assert not any('floor_candidate' in r or r['suspected_fall'] for r in moving)


def test_shared_ground_anchors_keep_identity_and_reject_occlusion():
    from fall_service import PoseBatch, ground_people_result
    batch=PoseBatch();batch.append(create_pose(.5,0.))
    batch.detections=[dict(pose_index=0,bbox=[.2,.1,.6,.85],confidence=.9)]
    tracks=[dict(pose_index=0,person_id=7,state='CHECKING')]
    people=ground_people_result(batch,tracks,640,480)
    assert len(people)==1 and people[0]['track_id']==7
    assert people[0]['keypoint_roles']==['ankle','ankle']
    batch[0][28]['visibility']=.1
    assert ground_people_result(batch,tracks,640,480)==[]
    batch[0][28]['visibility']=.99;tracks[0]['state']='AMBIGUOUS_PERSON'
    assert ground_people_result(batch,tracks,640,480)==[]


if __name__ == "__main__":
    print("=== RUNNING FALL DETECTOR V2 VALIDATION SUITE ===")
    test_positive_fall()
    test_negative_bending()
    test_negative_sitting()
    test_negative_slow_lying()
    test_already_lying_requires_ground_confirmation()
    test_floor_posture_brief_or_occluded_does_not_accumulate()
    test_lower_body_already_lying_and_folded_legs()
    test_moving_camera_suppresses_transition_not_ground_candidate()
    test_overlay_uses_one_pose_inference_and_keeps_partial_people_boxes()
    test_overlay_cache_keeps_capture_time_and_new_empty_frame_clears_boxes()
    test_overlay_floor_state_matches_existing_fall_suspicion_and_clears()
    test_transition_does_not_remain_latched_without_the_person()
    test_cropped_bent_leg_tracks_same_person_and_requires_depth()
    test_static_floor_timer_survives_a_change_of_observable_body_model()
    test_end_on_body_is_a_depth_candidate_only_and_never_an_rgb_fall()
    test_shared_ground_anchors_keep_identity_and_reject_occlusion()
    print("=== ALL 16 FALL DETECTION / SHARED POSE TEST SCENARIOS PASSED ===")
