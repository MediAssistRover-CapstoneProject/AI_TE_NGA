"""Synthetic motion tests only: NOT clinical performance validation."""
import math
from seizure_motion import SeizureMotionDetector


def pose(t, movement='both'):
    pts = [{'x':0.5,'y':0.5,'visibility':0.99} for _ in range(33)]
    pts[11].update(x=0.35,y=0.40)
    pts[12].update(x=0.65,y=0.40)
    # Right/left wrist positions relative to stationary shoulders.
    s = math.sin(2 * math.pi * 1.7 * t)
    l = 0.08 * s if movement in ('both', 'left') else 0.0
    r = 0.08 * math.sin(2 * math.pi * 1.7 * t + 0.2) if movement == 'both' else 0.0
    if movement == 'slow':
        l = 0.08 * math.sin(2 * math.pi * 0.25 * t)
        r = 0.08 * math.sin(2 * math.pi * 0.25 * t + 0.2)
    pts[15].update(x=0.20 + l, y=0.68)
    pts[16].update(x=0.80 + r, y=0.68)
    return pts


def scenario(kind, secs=11.0, fps=15):
    detector = SeizureMotionDetector()
    triggered = []
    phases = set()
    for i in range(int(secs*fps)):
        t=i/fps
        status, event = detector.update(pose(t,kind),t)
        phases.add(status['phase'])
        if event: triggered.append(event)
    return triggered, phases


events, phases = scenario('both')
assert len(events) == 1, ('expected one demo event', phases, events)
assert events[0]['event_type'] == 'SUSPECTED_SEIZURE'
assert events[0]['confidence'] is None
assert events[0]['simulation'] is True
assert not scenario('left')[0], 'One moving arm should not trigger default rule'
assert not scenario('none')[0], 'No movement should not trigger'
assert not scenario('slow')[0], 'Slow ordinary movement should not trigger'
print('SEIZURE_MOTION_TEST_PASS (synthetic bilateral/negative controls)')
