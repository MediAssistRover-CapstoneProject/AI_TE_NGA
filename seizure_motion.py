"""Proof-of-concept detector of prolonged bilateral rhythmic arm movement.

NOT a medical seizure diagnosis or life-safety detector. Does not observe brain
activity, consciousness, respiration or all seizure types. Use for staged tests.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import math
import time
import uuid

import numpy as np


class SeizureMotionDetector:
    """Look for repeated rhythmic wrist movement relative to a visible shoulder frame.

    Require both arms to show rhythmic motion for a sustained interval.
    This is intentionally conservative, and will miss many seizure presentations.
    """

    WINDOW_SECONDS = 6.0
    CONFIRM_SECONDS = 1.5
    COOLDOWN_SECONDS = 60.0
    MIN_FPS = 8.0
    MIN_VISIBILITY = 0.60
    MIN_SHOULDER_WIDTH = 0.08  # normalized image width
    MIN_RELATIVE_AMPLITUDE = 0.16  # wrist movement / shoulder width
    MIN_SPECTRAL_PEAK_SHARE = 0.22
    MIN_VALID_RATIO = 0.80
    FREQ_MIN_HZ = 0.8
    FREQ_MAX_HZ = 3.0

    def __init__(self):
        self.history = deque(maxlen=280)
        self.candidate_since = None
        self.cooldown_until = 0.0
        self.last_event = None
        self.last_metrics = {}

    def _point(self, landmarks, idx):
        if not landmarks or len(landmarks) <= idx:
            return None
        pt = landmarks[idx]
        # Accept original MediaPipe objects or JSON-serialized landmark dicts.
        if isinstance(pt, dict):
            x, y = pt.get('x'), pt.get('y')
            visibility = pt.get('visibility')
        else:
            x, y = getattr(pt, 'x', None), getattr(pt, 'y', None)
            visibility = getattr(pt, 'visibility', None)
        if x is None or y is None or visibility is None:
            return None
        try:
            x, y, visibility = float(x), float(y), float(visibility)
        except (ValueError, TypeError):
            return None
        if not all(map(math.isfinite, (x, y, visibility))):
            return None
        if visibility < self.MIN_VISIBILITY:
            return None
        return (x, y)

    def _observation(self, landmarks):
        ls = self._point(landmarks, 11)
        rs = self._point(landmarks, 12)
        lw = self._point(landmarks, 15)
        rw = self._point(landmarks, 16)
        if not ls or not rs:
            return None
        dx, dy = rs[0] - ls[0], rs[1] - ls[1]
        width = math.hypot(dx, dy)
        if width < self.MIN_SHOULDER_WIDTH:
            return None
        ux, uy = dx / width, dy / width

        def wrist_coords(shoulder, wrist):
            if wrist is None:
                return (float('nan'), float('nan'))
            vx, vy = wrist[0] - shoulder[0], wrist[1] - shoulder[1]
            # Body-relative coordinates, normalized by shoulder width.
            return ((vx * ux + vy * uy) / width,
                    (-vx * uy + vy * ux) / width)

        return (*wrist_coords(ls, lw), *wrist_coords(rs, rw))

    def _channel(self, times, values, end):
        ts = np.asarray(times, dtype=np.float64)
        xs = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(xs)
        if valid.mean() < self.MIN_VALID_RATIO or valid.sum() < 40:
            return None
        tv, xv = ts[valid], xs[valid]
        if np.max(np.diff(tv)) > 0.55:
            return None
        if tv[-1] - tv[0] < self.WINDOW_SECONDS - 0.75:
            return None

        dt = np.diff(tv)
        dt = dt[dt > 0.001]
        if not len(dt):
            return None
        fps = min(24.0, 1.0 / float(np.median(dt)))
        if fps < self.MIN_FPS:
            return None
        # Uniformly sampled for frequency analysis; avoid video timestamps
        # derived from non-realtime file playback.
        n = int((tv[-1] - tv[0]) * fps)
        if n < 40:
            return None
        grid = np.linspace(tv[0], tv[-1], n)
        arr = np.interp(grid, tv, xv)
        # Remove 1.2s moving baseline (slow repositioning/waving).
        span = max(3, int(round(fps * 1.2)))
        if span % 2 == 0:
            span += 1
        padding = span // 2
        smooth = np.convolve(
            np.pad(arr, (padding, padding), mode='edge'),
            np.ones(span) / span, mode='valid'
        )
        signal = arr - smooth
        signal = signal - float(np.mean(signal))
        fft = np.fft.rfft(signal * np.hanning(n))
        freq = np.fft.rfftfreq(n, d=float(grid[1] - grid[0]))
        band = (freq >= self.FREQ_MIN_HZ) & (freq <= self.FREQ_MAX_HZ)
        context = (freq >= 0.3) & (freq <= min(0.45 * fps, 5.0))
        if not np.any(band):
            return None
        power = np.abs(fft) ** 2
        peak_idx = np.flatnonzero(band)[np.argmax(power[band])]
        peak_share = float(power[peak_idx] / (power[context].sum() + 1e-12))
        # Correct for Hanning coherent gain, approximate sine amplitude.
        amplitude = float(2 * np.abs(fft[peak_idx]) / (np.hanning(n).sum() + 1e-12))
        if amplitude < self.MIN_RELATIVE_AMPLITUDE:
            return None
        if peak_share < self.MIN_SPECTRAL_PEAK_SHARE:
            return None
        return {
            'frequency_hz': round(float(freq[peak_idx]), 2),
            'amplitude_relative_shoulder': round(amplitude, 3),
            'peak_share': round(peak_share, 3),
            'sample_fps': round(fps, 1),
        }

    def _evaluate(self):
        if len(self.history) < 40:
            return {}, False
        samples = list(self.history)
        times = [entry[0] for entry in samples]
        if times[-1] - times[0] < self.WINDOW_SECONDS - 0.25:
            return {}, False

        signals = [[entry[k] for entry in samples] for k in range(1, 5)]
        features = [self._channel(times, vals, times[-1]) for vals in signals]
        left = max((x for x in features[:2] if x is not None),
                   key=lambda x: x['amplitude_relative_shoulder'], default=None)
        right = max((x for x in features[2:] if x is not None),
                    key=lambda x: x['amplitude_relative_shoulder'], default=None)
        metrics = {
            'left_arm': left,
            'right_arm': right,
            'window_seconds': round(times[-1] - times[0], 2),
            'rule': 'bilateral_repetitive_motion_only',
        }
        # Different rhythmic frequencies may be exercise/motion artifacts.
        rhythm = (
            left is not None and right is not None
            and abs(left['frequency_hz'] - right['frequency_hz']) <= 0.55
        )
        return metrics, rhythm

    def status(self, phase, now=None):
        if now is None:
            now = time.monotonic()
        return {
            'enabled': True,
            'phase': phase,
            'algorithm': 'bilateral_rhythmic_motion_poc_v1',
            'clinical_diagnosis': False,
            'simulation': True,
            'analysis_window_seconds': self.WINDOW_SECONDS,
            'confirmation_seconds': self.CONFIRM_SECONDS,
            'cooldown_remaining_seconds': round(max(0.0, self.cooldown_until - now), 1),
            'metrics': self.last_metrics,
        }

    def update(self, landmarks, now=None):
        if now is None:
            now = time.monotonic()
        observation = self._observation(landmarks)
        if observation is None:
            self.candidate_since = None
            self.history.clear()
            self.last_metrics = {}
            return self.status('LOW_VISIBILITY' if landmarks else 'NO_PERSON', now), None

        self.history.append((float(now), *observation))
        while self.history and now - self.history[0][0] > self.WINDOW_SECONDS + 0.30:
            self.history.popleft()
        if now < self.cooldown_until:
            return self.status('COOLDOWN', now), None

        if len(self.history) >= 2:
            elapsed = self.history[-1][0] - self.history[0][0]
            fps = (len(self.history) - 1) / max(elapsed, 0.001)
            if fps < self.MIN_FPS and elapsed > 2.0:
                self.candidate_since = None
                return self.status('LOW_FPS', now), None

        metrics, sustained_rhythm = self._evaluate()
        self.last_metrics = metrics
        if not sustained_rhythm:
            self.candidate_since = None
            return self.status('OBSERVING', now), None
        if self.candidate_since is None:
            self.candidate_since = now
            return self.status('RHYTHMIC_MOTION_CHECKING', now), None
        if now - self.candidate_since < self.CONFIRM_SECONDS:
            return self.status('RHYTHMIC_MOTION_CHECKING', now), None

        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'SUSPECTED_SEIZURE',
            'source': 'CAMERA_AI',
            'detected_at': datetime.now(timezone.utc).isoformat(),
            'algorithm': 'bilateral_rhythmic_motion_poc_v1',
            'confidence': None,
            'simulation': True,
            'clinical_diagnosis': False,
            'requires_staff_verification': True,
            'evidence_summary': metrics,
        }
        self.last_event = event
        self.cooldown_until = now + self.COOLDOWN_SECONDS
        self.candidate_since = None
        return self.status('SUSPECTED_SEIZURE', now), event
