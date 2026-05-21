"""Map hand landmarks to drone Commands.

Pure functions + a small debounce state machine. No drone, no camera, no
MediaPipe imports — easy to unit test.

MediaPipe landmark indices used here:
    0  wrist
    4  thumb tip          (3 thumb IP, 2 thumb MCP)
    8  index tip          (6 index PIP)
    12 middle tip         (10 middle PIP)
    16 ring tip           (14 ring PIP)
    20 pinky tip          (18 pinky PIP)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Actions
ACTION_NONE = "none"
ACTION_TAKEOFF = "takeoff"
ACTION_LAND = "land"
ACTION_ESTOP = "estop"
ACTION_FLY = "fly"


@dataclass
class Command:
    action: str = ACTION_NONE
    pitch: float = 0.0
    roll: float = 0.0
    throttle: float = 0.0
    yaw: float = 0.0


def fingers_up(landmarks: np.ndarray) -> List[bool]:
    """Return [thumb, index, middle, ring, pinky] up/down booleans.

    Thumb compares x (it sticks out sideways); others compare y (tip above PIP
    in image coords means smaller y).
    """
    lm = landmarks
    # Thumb: tip vs IP joint on x. Direction depends on which hand; we use
    # absolute distance + a sign relative to wrist for robustness.
    thumb_up = abs(lm[4, 0] - lm[2, 0]) > 0.04 and lm[4, 1] < lm[2, 1] + 0.02
    index_up = lm[8, 1] < lm[6, 1] - 0.02
    middle_up = lm[12, 1] < lm[10, 1] - 0.02
    ring_up = lm[16, 1] < lm[14, 1] - 0.02
    pinky_up = lm[20, 1] < lm[18, 1] - 0.02
    return [thumb_up, index_up, middle_up, ring_up, pinky_up]


def classify_static_gesture(landmarks: np.ndarray) -> str:
    """Return one of: 'open_palm', 'fist', 'thumbs_up', 'index_only', 'other'."""
    f = fingers_up(landmarks)
    thumb, index, middle, ring, pinky = f
    if all([thumb, index, middle, ring, pinky]):
        return "open_palm"
    if not any([index, middle, ring, pinky]) and not thumb:
        return "fist"
    if thumb and not any([index, middle, ring, pinky]):
        return "thumbs_up"
    if index and not any([middle, ring, pinky]):
        return "index_only"
    return "other"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class MapperConfig:
    deadzone: float = 0.12       # fraction of frame around center that maps to 0
    max_speed: int = 50          # codrone speeds are in [-100, 100]
    hold_frames: int = 12        # ~0.4s at 30fps to trigger one-shot gestures
    fly_hold_frames: int = 3     # smoother enter into continuous flight


@dataclass
class GestureMapper:
    config: MapperConfig = field(default_factory=MapperConfig)

    # Internal debounce state
    _last_gesture: str = "other"
    _gesture_streak: int = 0
    _armed_actions: set = field(default_factory=set)  # actions already fired this hold

    def reset(self) -> None:
        self._last_gesture = "other"
        self._gesture_streak = 0
        self._armed_actions.clear()

    def update(self, landmarks: Optional[np.ndarray]) -> Command:
        if landmarks is None:
            # No hand: caller decides what to do (auto-land timer lives in controller).
            self._last_gesture = "other"
            self._gesture_streak = 0
            return Command(action=ACTION_NONE)

        gesture = classify_static_gesture(landmarks)
        if gesture == self._last_gesture:
            self._gesture_streak += 1
        else:
            self._last_gesture = gesture
            self._gesture_streak = 1
            self._armed_actions.clear()

        cfg = self.config

        # One-shot gestures: fire once after hold_frames, then re-arm only after
        # the user changes gesture (cleared by the streak reset above).
        if gesture == "open_palm" and self._gesture_streak >= cfg.hold_frames \
                and ACTION_TAKEOFF not in self._armed_actions:
            self._armed_actions.add(ACTION_TAKEOFF)
            return Command(action=ACTION_TAKEOFF)
        if gesture == "fist" and self._gesture_streak >= cfg.hold_frames \
                and ACTION_LAND not in self._armed_actions:
            self._armed_actions.add(ACTION_LAND)
            return Command(action=ACTION_LAND)
        if gesture == "thumbs_up" and self._gesture_streak >= cfg.hold_frames \
                and ACTION_ESTOP not in self._armed_actions:
            self._armed_actions.add(ACTION_ESTOP)
            return Command(action=ACTION_ESTOP)

        # Continuous flight: index finger up only.
        if gesture == "index_only" and self._gesture_streak >= cfg.fly_hold_frames:
            return self._fly_command(landmarks)

        return Command(action=ACTION_NONE)

    def _fly_command(self, lm: np.ndarray) -> Command:
        cfg = self.config
        wrist = lm[0]
        # Wrist offset from center, in [-1, 1] roughly (normalized landmark space).
        dx = (wrist[0] - 0.5) * 2.0
        dy = (wrist[1] - 0.5) * 2.0

        def axis(value: float) -> float:
            if abs(value) < cfg.deadzone:
                return 0.0
            # Scale linearly from deadzone..1 → 0..max_speed
            sign = 1.0 if value > 0 else -1.0
            mag = (abs(value) - cfg.deadzone) / (1.0 - cfg.deadzone)
            return sign * _clamp(mag, 0.0, 1.0) * cfg.max_speed

        roll = axis(dx)
        # Frame y grows downward → invert so "hand up" raises drone.
        throttle = -axis(dy)

        # Pitch: angle of index finger (wrist→tip) relative to vertical.
        tip = lm[8]
        vx, vy = tip[0] - wrist[0], tip[1] - wrist[1]
        # If finger points straight up, vy is strongly negative → pitch ~ 0.
        # Tilting tip forward (toward camera top, smaller y) doesn't help — instead
        # we use the wrist depth coordinate (z) for forward/back; negative z is
        # closer to the camera in MediaPipe space.
        pitch_signal = -lm[0, 2]  # closer hand → positive pitch (forward)
        pitch = axis(_clamp(pitch_signal * 4.0, -1.0, 1.0))

        return Command(
            action=ACTION_FLY,
            roll=roll,
            throttle=throttle,
            pitch=pitch,
            yaw=0.0,
        )
