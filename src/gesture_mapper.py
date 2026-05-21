"""Two-handed RC-stick gesture mapper.

Design:
    The user's two hands act like the two sticks on a standard "Mode 2"
    drone RC transmitter:

        LEFT hand (left half of frame)  →  Up/Down  (Y)  +  Turn  (X)
        RIGHT hand (right half of frame)→  Fwd/Back (Y)  +  Left/Right (X)

    Each stick reads only the 2-D image position of the hand inside its own
    "stick box" (a neutral point with a fixed range around it). Because each
    hand drives only its own axes, moving one hand can never accidentally
    change the other axes — the system is fully decoupled.

    Action gestures:
        BOTH hands open palm  → takeoff
        BOTH hands closed fist→ land
        EITHER thumb down     → emergency stop (single-hand for emergencies)

    Anything else with two hands visible = continuous flight.
    Anything with fewer than two hands and no action gesture = hover.

Pure logic — no MediaPipe / drone imports — so it's easy to unit test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .smoothing import OneEuroFilter


# --- Actions -----------------------------------------------------------------

ACTION_NONE = "none"
ACTION_TAKEOFF = "takeoff"
ACTION_LAND = "land"
ACTION_ESTOP = "estop"
ACTION_FLY = "fly"


# --- Public types ------------------------------------------------------------

@dataclass
class HandObservation:
    """One detected hand in a frame."""
    landmarks: np.ndarray  # (21, 3), normalized
    gesture: str = ""
    score: float = 0.0


@dataclass
class Command:
    action: str = ACTION_NONE
    pitch: float = 0.0    # forward (+) / back (-)
    roll: float = 0.0     # right (+) / left (-)
    throttle: float = 0.0 # up (+) / down (-)
    yaw: float = 0.0      # turn right (+) / left (-)


@dataclass
class MapperConfig:
    max_speed: int = 50
    expo: float = 0.4
    deadzone: float = 0.18         # fraction of stick_range — small wobbles → 0
    hold_frames: int = 10          # ~0.33 s @ 30 fps for takeoff/land
    estop_hold_frames: int = 12    # E-stop needs a deliberate hold too
    min_gesture_score: float = 0.55

    # Stick layout in frame-normalized coords. Default: left stick centered in
    # the left half, right stick centered in the right half.
    left_neutral_x: float = 0.25
    left_neutral_y: float = 0.50
    right_neutral_x: float = 0.75
    right_neutral_y: float = 0.50
    stick_range: float = 0.20      # half-range of each axis (fraction of frame)


# --- Internals ---------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _expo(v: float, k: float) -> float:
    sign = 1.0 if v >= 0 else -1.0
    m = min(abs(v), 1.0)
    return sign * ((1 - k) * m + k * m * m * m)


# --- Mapper ------------------------------------------------------------------

@dataclass
class GestureMapper:
    config: MapperConfig = field(default_factory=MapperConfig)

    # Action debounce state
    _last_action: str = ""
    _action_streak: int = 0
    _fired: set = field(default_factory=set)

    # Output smoothers
    _pitch_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.2, beta=0.04))
    _roll_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.2, beta=0.04))
    _throttle_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.2, beta=0.04))
    _yaw_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.0, beta=0.03))

    def reset(self) -> None:
        self._last_action = ""
        self._action_streak = 0
        self._fired.clear()
        for f in (self._pitch_f, self._roll_f, self._throttle_f, self._yaw_f):
            f.reset()

    # -- main entry --
    def update(
        self,
        hands: List[HandObservation],
        timestamp: Optional[float] = None,
    ) -> Command:
        if not hands:
            self._reset_action_streak()
            return Command(action=ACTION_NONE)

        # Sort by wrist X so index 0 is the user's LEFT-side hand on screen
        # (we already mirror the frame in main, so this matches the user's POV).
        ordered = sorted(hands, key=lambda h: float(h.landmarks[0, 0]))

        fired, pending = self._detect_action(ordered)
        if fired:
            return Command(action=fired)

        # User is holding an action gesture but the hold hasn't fired yet —
        # don't fly through it. Just hover.
        if pending:
            return Command(action=ACTION_NONE)

        if len(ordered) >= 2:
            return self._fly(left=ordered[0], right=ordered[-1], t=timestamp)

        # Single hand visible without an action gesture → hover (no input).
        return Command(action=ACTION_NONE)

    # -- action detection --
    # Returns (fired_action, pending_action_bool). `fired` is non-empty only on
    # the frame the hold completes. `pending` is True while the user is holding
    # a recognized action gesture but the hold hasn't elapsed yet — caller
    # should hover, not fly, during that window.
    def _detect_action(self, hands: List[HandObservation]):
        cfg = self.config

        # Any confident thumb-down → E-stop
        for h in hands:
            if h.gesture == "Thumb_Down" and h.score >= cfg.min_gesture_score:
                return self._track_action(ACTION_ESTOP, cfg.estop_hold_frames), True

        # Takeoff / land require BOTH hands to agree
        if len(hands) >= 2 and all(h.score >= cfg.min_gesture_score for h in hands[:2]):
            gestures = {hands[0].gesture, hands[1].gesture}
            if gestures == {"Open_Palm"}:
                return self._track_action(ACTION_TAKEOFF, cfg.hold_frames), True
            if gestures == {"Closed_Fist"}:
                return self._track_action(ACTION_LAND, cfg.hold_frames), True

        self._reset_action_streak()
        return "", False

    def _track_action(self, action: str, hold: int) -> str:
        if action == self._last_action:
            self._action_streak += 1
        else:
            self._last_action = action
            self._action_streak = 1
            self._fired.clear()
        if self._action_streak >= hold and action not in self._fired:
            self._fired.add(action)
            return action
        return ""

    def _reset_action_streak(self) -> None:
        self._last_action = ""
        self._action_streak = 0
        self._fired.clear()

    # -- fly mode --
    def _fly(
        self,
        left: HandObservation,
        right: HandObservation,
        t: Optional[float],
    ) -> Command:
        cfg = self.config

        # Use the palm center (mean of wrist + 4 MCPs) as the reference point —
        # it's much more stable than the wrist on its own.
        l_palm = left.landmarks[[0, 5, 9, 13, 17]].mean(axis=0)
        r_palm = right.landmarks[[0, 5, 9, 13, 17]].mean(axis=0)

        # Left stick: throttle (Y) + yaw (X)
        left_dx = (float(l_palm[0]) - cfg.left_neutral_x) / cfg.stick_range
        left_dy = (float(l_palm[1]) - cfg.left_neutral_y) / cfg.stick_range
        throttle_n = self._shape(-left_dy)  # hand up (smaller y) → throttle up
        yaw_n = self._shape(left_dx)        # hand right → turn right

        # Right stick: pitch (Y) + roll (X)
        right_dx = (float(r_palm[0]) - cfg.right_neutral_x) / cfg.stick_range
        right_dy = (float(r_palm[1]) - cfg.right_neutral_y) / cfg.stick_range
        pitch_n = self._shape(-right_dy)    # hand up → forward
        roll_n = self._shape(right_dx)      # hand right → fly right

        return Command(
            action=ACTION_FLY,
            pitch=self._pitch_f(pitch_n * cfg.max_speed, t),
            roll=self._roll_f(roll_n * cfg.max_speed, t),
            throttle=self._throttle_f(throttle_n * cfg.max_speed, t),
            yaw=self._yaw_f(yaw_n * cfg.max_speed, t),
        )

    def _shape(self, v: float) -> float:
        cfg = self.config
        v = _clamp(v, -1.0, 1.0)
        if abs(v) < cfg.deadzone:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        rescaled = (abs(v) - cfg.deadzone) / (1.0 - cfg.deadzone)
        return sign * _expo(_clamp(rescaled, 0.0, 1.0), cfg.expo)
