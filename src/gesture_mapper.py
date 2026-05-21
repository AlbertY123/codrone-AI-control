"""Map recognized hand state to drone Commands.

Uses the pretrained gesture label (Open_Palm / Closed_Fist / Thumb_Up /
Pointing_Up / Victory / ILoveYou / Thumb_Down / None) for high-level actions,
and uses palm-center + palm-tilt geometry for continuous flight in "Pointing_Up"
mode.

Pure: no MediaPipe / camera / drone imports. Easy to unit test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .smoothing import OneEuroFilter

# Actions
ACTION_NONE = "none"
ACTION_TAKEOFF = "takeoff"
ACTION_LAND = "land"
ACTION_ESTOP = "estop"
ACTION_FLY = "fly"
ACTION_FLIP = "flip"  # reserved; not auto-fired


@dataclass
class Command:
    action: str = ACTION_NONE
    pitch: float = 0.0
    roll: float = 0.0
    throttle: float = 0.0
    yaw: float = 0.0


@dataclass
class MapperConfig:
    deadzone: float = 0.10          # fraction of frame around center → 0 output
    max_speed: int = 50             # codrone_edu speed range is [-100, 100]
    expo: float = 0.35              # 0 = linear, higher = softer near center
    hold_frames: int = 10           # ~0.33 s @ 30 fps for one-shot gestures
    fly_hold_frames: int = 3        # smaller hold to enter continuous mode
    estop_hold_frames: int = 18     # E-stop deliberately requires a longer hold
    min_gesture_score: float = 0.55 # ignore low-confidence gesture predictions
    yaw_gain: float = 1.4           # how aggressively hand roll maps to yaw
    # Forward/back pitch: derived from apparent hand size. The baseline is the
    # palm length (wrist → middle-finger MCP) in normalized image coords at the
    # user's "neutral" distance from the camera. Bigger than baseline ⇒ closer
    # to camera ⇒ forward. Smaller ⇒ farther ⇒ backward.
    pitch_baseline: float = 0.18
    pitch_gain: float = 6.0
    pitch_deadband: float = 0.15    # ignore deviations smaller than this (fraction)


# Gesture name → high-level action. "Pointing_Up" stays as continuous control,
# handled separately.
_ONE_SHOT_ACTIONS = {
    "Open_Palm": ACTION_TAKEOFF,
    "Closed_Fist": ACTION_LAND,
    "Thumb_Down": ACTION_ESTOP,
}


def _expo(v: float, k: float) -> float:
    """Symmetric expo curve: soft near zero, full near ±1."""
    sign = 1.0 if v >= 0 else -1.0
    m = min(abs(v), 1.0)
    return sign * ((1 - k) * m + k * m * m * m)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class GestureMapper:
    config: MapperConfig = field(default_factory=MapperConfig)

    # Debounce state
    _last_gesture: str = ""
    _streak: int = 0
    _fired: set = field(default_factory=set)

    # Output smoothers
    _pitch_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.2, beta=0.04))
    _roll_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.2, beta=0.04))
    _throttle_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.2, beta=0.04))
    _yaw_f: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(min_cutoff=1.0, beta=0.03))

    def reset(self) -> None:
        self._last_gesture = ""
        self._streak = 0
        self._fired.clear()
        for f in (self._pitch_f, self._roll_f, self._throttle_f, self._yaw_f):
            f.reset()

    # --- main entry ------------------------------------------------------------

    def update(
        self,
        landmarks: Optional[np.ndarray],
        gesture: str = "",
        score: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Command:
        # No hand → reset state, hand off to controller (it auto-lands).
        if landmarks is None:
            self._last_gesture = ""
            self._streak = 0
            self._fired.clear()
            return Command(action=ACTION_NONE)

        cfg = self.config

        # Treat low-confidence predictions as "no gesture" (still in flight mode if eligible).
        effective = gesture if score >= cfg.min_gesture_score else ""

        # Track streak for debouncing.
        if effective == self._last_gesture:
            self._streak += 1
        else:
            self._last_gesture = effective
            self._streak = 1
            self._fired.clear()

        # One-shot actions: open palm / fist / thumb-down → fire once after hold.
        if effective in _ONE_SHOT_ACTIONS:
            hold = cfg.estop_hold_frames if effective == "Thumb_Down" else cfg.hold_frames
            action = _ONE_SHOT_ACTIONS[effective]
            if self._streak >= hold and action not in self._fired:
                self._fired.add(action)
                return Command(action=action)
            return Command(action=ACTION_NONE)

        # Continuous control: hand visible (any non-action gesture) → fly.
        if self._streak >= cfg.fly_hold_frames:
            return self._fly_command(landmarks, timestamp)

        return Command(action=ACTION_NONE)

    # --- continuous control ----------------------------------------------------

    def _fly_command(self, lm: np.ndarray, t: Optional[float]) -> Command:
        cfg = self.config

        # Palm center = mean of wrist (0) and MCPs of index/middle/ring/pinky (5,9,13,17).
        palm = lm[[0, 5, 9, 13, 17]].mean(axis=0)
        # Offsets from frame center, in roughly [-1, 1].
        dx = (palm[0] - 0.5) * 2.0
        dy = (palm[1] - 0.5) * 2.0

        # Hand roll = angle of the line from pinky-MCP (17) to index-MCP (5).
        # When palm is held flat-vertical the line is roughly horizontal; tilting
        # the hand left or right rotates it. This drives yaw.
        v = lm[5] - lm[17]
        roll_angle = math.atan2(v[1], v[0])  # radians; 0 ≈ flat (index right of pinky)
        # Normalize so flat hand ~ 0.
        norm_yaw = _clamp(roll_angle / (math.pi / 2), -1.0, 1.0) * cfg.yaw_gain
        norm_yaw = _clamp(norm_yaw, -1.0, 1.0)

        # Forward/back pitch from apparent palm size in the image.
        # Wrist-z in MediaPipe is unreliable on its own; the projected palm
        # length is much more stable. Bigger than baseline → closer → forward.
        palm_len = float(np.linalg.norm(lm[9, :2] - lm[0, :2]))
        pitch_raw = (palm_len - cfg.pitch_baseline) / cfg.pitch_baseline
        if abs(pitch_raw) < cfg.pitch_deadband:
            pitch_signal = 0.0
        else:
            sign = 1.0 if pitch_raw > 0 else -1.0
            pitch_signal = _clamp(
                sign * (abs(pitch_raw) - cfg.pitch_deadband) * cfg.pitch_gain,
                -1.0, 1.0,
            )

        roll_norm = self._shape(dx)
        throttle_norm = self._shape(-dy)  # invert: hand up (small y) → throttle up
        # pitch_signal is already deadbanded above, so use zero deadzone here.
        pitch_norm = self._shape(pitch_signal, deadzone=0.0)

        roll_out = self._roll_f(roll_norm * cfg.max_speed, t)
        throttle_out = self._throttle_f(throttle_norm * cfg.max_speed, t)
        pitch_out = self._pitch_f(pitch_norm * cfg.max_speed, t)
        yaw_out = self._yaw_f(self._shape(norm_yaw, deadzone=cfg.deadzone) * cfg.max_speed, t)

        return Command(
            action=ACTION_FLY,
            pitch=pitch_out,
            roll=roll_out,
            throttle=throttle_out,
            yaw=yaw_out,
        )

    def _shape(self, v: float, deadzone: Optional[float] = None) -> float:
        cfg = self.config
        dz = cfg.deadzone if deadzone is None else deadzone
        if abs(v) < dz:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        # Rescale (dz, 1) → (0, 1) then apply expo.
        rescaled = (abs(v) - dz) / (1.0 - dz)
        return sign * _expo(_clamp(rescaled, 0.0, 1.0), cfg.expo)
