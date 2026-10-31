import math
import time
import uuid
from datetime import datetime, timezone


class HandSOSDetector:
    """
    Signal for Help:
    1. Open palm, thumb outside
    2. Tuck thumb into palm, fingers open
    3. Fold fingers over thumb

    Heuristic demo. Not a validated gesture classifier.
    """

    OPEN_HOLD = 0.45
    TUCK_HOLD = 0.45
    TRAP_HOLD = 0.45
    TRANSITION_LIMIT = 4.0
    COOLDOWN = 5.0

    def __init__(self):
        self.phase = "WAIT_OPEN"
        self.hand = None
        self.hold_start = None
        self.deadline = 0.0
        self.cooldown_until = 0.0
        self.last_seen = None
        self.last_event = None
        self.debug = {}

    def reset(self):
        self.phase = "WAIT_OPEN"
        self.hand = None
        self.hold_start = None
        self.deadline = 0.0
        self.last_seen = None

    @staticmethod
    def _xy(p):
        return float(p["x"]), float(p["y"])

    def classify(self, hand):
        points = hand.get("landmarks") or []
        fingers = hand.get("fingers") or {}

        if len(points) != 21:
            return "UNKNOWN", {}

        try:
            # Wrist, thumb tip, index MCP,
            # middle MCP and pinky MCP.
            wrist = self._xy(points[0])
            thumb = self._xy(points[4])
            index = self._xy(points[5])
            middle = self._xy(points[9])
            pinky = self._xy(points[17])

            ax = pinky[0] - index[0]
            ay = pinky[1] - index[1]
            width2 = ax * ax + ay * ay

            bx = middle[0] - wrist[0]
            by = middle[1] - wrist[1]
            length2 = bx * bx + by * by

            if width2 < 0.00025 or length2 < 0.00025:
                return "UNKNOWN", {"reason": "HAND_TOO_SMALL"}

            # Horizontal position relative to palm width:
            # 0 = index knuckle, 1 = pinky knuckle.
            thumb_u = (
                (thumb[0] - index[0]) * ax
                + (thumb[1] - index[1]) * ay
            ) / width2

            # Position along wrist-to-knuckle direction.
            thumb_v = (
                (thumb[0] - wrist[0]) * bx
                + (thumb[1] - wrist[1]) * by
            ) / length2

            names = ["index", "middle", "ring", "pinky"]

            states = [
                fingers.get(k, {}).get("state", "UNKNOWN")
                for k in names
            ]

            open_count = states.count("OPEN")
            bent_count = states.count("BENT")

            fingers_open = (
                open_count >= 3 and bent_count == 0
            )
            fingers_closed = (
                bent_count >= 3 and open_count <= 1
            )

            thumb_out = thumb_u < -0.15

            # Thumb tip inside approximate palm area.
            thumb_tucked = (
                0.10 <= thumb_u <= 1.15
                and 0.10 <= thumb_v <= 1.45
            )

            debug = {
                "thumb_u": round(thumb_u, 3),
                "thumb_v": round(thumb_v, 3),
                "thumb_out": thumb_out,
                "thumb_tucked": thumb_tucked,
                "open_fingers": open_count,
                "bent_fingers": bent_count,
            }

            if fingers_open and thumb_out:
                return "OPEN_PALM", debug

            if fingers_open and thumb_tucked:
                return "TUCKED_THUMB", debug

            if fingers_closed:
                # Thumb may be hidden under the other
                # fingers. This confirms a closed-hand
                # pose, not optical proof of thumb trapping.
                return "FINGERS_CLOSED", debug

            return "OTHER", debug

        except (ValueError, TypeError, KeyError, IndexError):
            return "UNKNOWN", {"reason": "INVALID_LANDMARKS"}

    def status(self, now=None):
        if now is None:
            now = time.monotonic()

        instructions = {
            "WAIT_OPEN": "B1: Xoe ban tay, ngon cai mo ra ngoai",
            "HOLD_OPEN": "B1: Giu ban tay mo",
            "WAIT_TUCK": "B2: Gap ngon cai vao long ban tay",
            "HOLD_TUCK": "B2: Giu ngon cai gap, 4 ngon van mo",
            "WAIT_TRAP": "B3: Gap 4 ngon xuong bao ngon cai",
            "HOLD_TRAP": "B3: Giu tu the nam tay",
            "TRIGGERED": "DA NHAN DIEN SIGNAL FOR HELP",
            "COOLDOWN": "Tam nghi de tranh lap canh bao",
        }

        step = 1
        if self.phase in ("WAIT_TUCK", "HOLD_TUCK"):
            step = 2
        elif self.phase in (
            "WAIT_TRAP", "HOLD_TRAP", "TRIGGERED"
        ):
            step = 3

        remaining = 0.0
        if self.phase in (
            "WAIT_TUCK", "HOLD_TUCK",
            "WAIT_TRAP", "HOLD_TRAP"
        ):
            remaining = max(0, self.deadline - now)

        return {
            "enabled": True,
            "gesture": "SIGNAL_FOR_HELP",
            "phase": self.phase,
            "step": step,
            "hand": self.hand,
            "instruction": instructions.get(self.phase, ""),
            "remaining_sec": round(remaining, 1),
            "debug": dict(self.debug),
        }

    def update(self, hands, now=None):
        if now is None:
            now = time.monotonic()

        if now < self.cooldown_until:
            self.phase = "COOLDOWN"
            return self.status(now), None

        if self.phase in ("COOLDOWN", "TRIGGERED"):
            self.reset()

        if self.deadline and now > self.deadline:
            self.reset()

        observations = []

        for hand in hands or []:
            shape, info = self.classify(hand)
            observations.append((hand, shape, info))

        # Start with one open palm.
        if self.phase == "WAIT_OPEN":
            selected = next(
                (
                    obs for obs in observations
                    if obs[1] == "OPEN_PALM"
                ),
                None
            )

            if selected is None:
                self.debug = (
                    observations[0][2]
                    if observations else {}
                )
                return self.status(now), None

            hand, shape, info = selected
            self.hand = hand.get("hand", "Unknown")
            self.last_seen = now
            self.debug = info
            self.phase = "HOLD_OPEN"
            self.hold_start = now

            return self.status(now), None

        # Keep monitoring the same hand label.
        selected = next(
            (
                obs for obs in observations
                if obs[0].get("hand", "Unknown")
                == self.hand
            ),
            None
        )

        if selected is None:
            if (
                self.last_seen is not None
                and now - self.last_seen > 0.7
            ):
                self.reset()

            return self.status(now), None

        hand, shape, info = selected
        self.last_seen = now
        self.debug = info

        if self.phase == "HOLD_OPEN":
            if shape != "OPEN_PALM":
                self.reset()
            elif now - self.hold_start >= self.OPEN_HOLD:
                self.phase = "WAIT_TUCK"
                self.hold_start = None
                self.deadline = now + self.TRANSITION_LIMIT

        elif self.phase == "WAIT_TUCK":
            if shape == "TUCKED_THUMB":
                self.phase = "HOLD_TUCK"
                self.hold_start = now

        elif self.phase == "HOLD_TUCK":
            if shape != "TUCKED_THUMB":
                self.phase = "WAIT_TUCK"
                self.hold_start = None
            elif now - self.hold_start >= self.TUCK_HOLD:
                self.phase = "WAIT_TRAP"
                self.hold_start = None
                self.deadline = now + self.TRANSITION_LIMIT

        elif self.phase == "WAIT_TRAP":
            if shape == "FINGERS_CLOSED":
                self.phase = "HOLD_TRAP"
                self.hold_start = now

        elif self.phase == "HOLD_TRAP":
            if shape != "FINGERS_CLOSED":
                self.phase = "WAIT_TRAP"
                self.hold_start = None
            elif now - self.hold_start >= self.TRAP_HOLD:
                event = {
                    "event_id": str(uuid.uuid4()),
                    "event_type": "GESTURE_SOS",
                    "gesture_code": "SIGNAL_FOR_HELP",
                    "source": "CAMERA_AI",
                    "hand": self.hand,
                    "detected_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "algorithm": "signal_for_help_v1",
                    "confidence": None,
                    "requires_staff_verification": True,
                    "simulation": True,
                }

                self.last_event = event
                self.phase = "TRIGGERED"
                self.cooldown_until = now + self.COOLDOWN
                self.deadline = 0.0

                return self.status(now), event

        return self.status(now), None
