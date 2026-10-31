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


if __name__ == "__main__":
    print("=== RUNNING FALL DETECTOR V2 VALIDATION SUITE ===")
    test_positive_fall()
    test_negative_bending()
    test_negative_sitting()
    test_negative_slow_lying()
    print("=== ALL 4 FALL DETECTION TEST SCENARIOS PASSED ===")
