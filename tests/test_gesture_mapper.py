"""Unit tests for the gesture mapper and smoothing.

The mapper now relies on the pretrained gesture *name* rather than re-running
a finger-up heuristic, so the tests inject gesture labels directly.
"""
from __future__ import annotations

import math

import numpy as np

from src.gesture_mapper import (
    ACTION_ESTOP,
    ACTION_FLY,
    ACTION_LAND,
    ACTION_NONE,
    ACTION_TAKEOFF,
    GestureMapper,
    MapperConfig,
)
from src.smoothing import OneEuroFilter, ScalarEMA


def _hand_at(wx: float = 0.5, wy: float = 0.5, wz: float = 0.0) -> np.ndarray:
    """Build a minimally-valid 21x3 landmark array centered at (wx, wy)."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (wx, wy, wz)
    # Place finger MCPs around the wrist; the mapper averages them for palm center.
    # MCPs sit roughly one palm-length above the wrist so the wrist→middle-MCP
    # distance matches MapperConfig.pitch_baseline (= 0.18), keeping the
    # forward/back signal at zero in tests unless we explicitly change palm size.
    lm[5] = (wx + 0.05, wy - 0.18, 0.0)   # index MCP
    lm[9] = (wx + 0.00, wy - 0.18, 0.0)   # middle MCP
    lm[13] = (wx - 0.03, wy - 0.18, 0.0)  # ring MCP
    lm[17] = (wx - 0.06, wy - 0.18, 0.0)  # pinky MCP
    return lm


def _hand_scaled(wx: float, wy: float, palm_len: float) -> np.ndarray:
    """Hand whose wrist→middle-MCP length matches `palm_len`."""
    lm = _hand_at(wx, wy)
    lm[9] = (wx, wy - palm_len, 0.0)
    return lm


def test_no_hand_returns_none():
    m = GestureMapper()
    assert m.update(None).action == ACTION_NONE


def test_takeoff_requires_hold():
    cfg = MapperConfig(hold_frames=4)
    m = GestureMapper(config=cfg)
    lm = _hand_at()
    for _ in range(cfg.hold_frames - 1):
        assert m.update(lm, "Open_Palm", 0.9).action == ACTION_NONE
    assert m.update(lm, "Open_Palm", 0.9).action == ACTION_TAKEOFF
    # Re-firing requires breaking the streak.
    assert m.update(lm, "Open_Palm", 0.9).action == ACTION_NONE


def test_low_score_does_not_trigger_oneshot():
    cfg = MapperConfig(hold_frames=2, min_gesture_score=0.7)
    m = GestureMapper(config=cfg)
    lm = _hand_at()
    for _ in range(10):
        # 0.4 < threshold → treated as "" gesture.
        assert m.update(lm, "Open_Palm", 0.4).action != ACTION_TAKEOFF


def test_land_after_takeoff():
    cfg = MapperConfig(hold_frames=2)
    m = GestureMapper(config=cfg)
    lm = _hand_at()
    m.update(lm, "Open_Palm", 0.9); m.update(lm, "Open_Palm", 0.9)
    m.update(lm, "Closed_Fist", 0.9)
    assert m.update(lm, "Closed_Fist", 0.9).action == ACTION_LAND


def test_estop_requires_longer_hold():
    cfg = MapperConfig(hold_frames=3, estop_hold_frames=8)
    m = GestureMapper(config=cfg)
    lm = _hand_at()
    # 7 frames of Thumb_Down: not enough yet.
    for _ in range(7):
        assert m.update(lm, "Thumb_Down", 0.9).action == ACTION_NONE
    assert m.update(lm, "Thumb_Down", 0.9).action == ACTION_ESTOP


def test_fly_mode_outputs_command_with_centered_hand():
    cfg = MapperConfig(deadzone=0.15, fly_hold_frames=1, max_speed=50)
    m = GestureMapper(config=cfg)
    # Pointing_Up triggers continuous control. Place wrist so that the palm
    # center (mean of wrist + 4 MCPs, each 0.18 above the wrist) lands at
    # frame-center (0.5, 0.5): wrist_y = 0.5 + 4*0.18/5 = 0.644.
    cmd = m.update(_hand_at(0.5, 0.644), "Pointing_Up", 0.9)
    assert cmd.action == ACTION_FLY
    assert abs(cmd.roll) < 1.0
    assert abs(cmd.throttle) < 1.0
    assert abs(cmd.pitch) < 1.0


def test_fly_mode_roll_right_when_hand_right():
    cfg = MapperConfig(deadzone=0.05, fly_hold_frames=1, max_speed=50, expo=0.0)
    m = GestureMapper(config=cfg)
    cmd = m.update(_hand_at(0.95, 0.5), "Pointing_Up", 0.9)
    assert cmd.action == ACTION_FLY
    assert cmd.roll > 0.0


def test_fly_mode_throttle_up_when_hand_high():
    cfg = MapperConfig(deadzone=0.05, fly_hold_frames=1, max_speed=50, expo=0.0)
    m = GestureMapper(config=cfg)
    cmd = m.update(_hand_at(0.5, 0.05), "Pointing_Up", 0.9)
    assert cmd.action == ACTION_FLY
    assert cmd.throttle > 0.0


def test_fly_mode_pitch_forward_when_hand_close():
    cfg = MapperConfig(deadzone=0.05, fly_hold_frames=1, max_speed=50, expo=0.0,
                       pitch_baseline=0.18, pitch_deadband=0.1, pitch_gain=4.0)
    m = GestureMapper(config=cfg)
    # palm_len = 0.30 (well above baseline 0.18) → hand close to camera → forward
    cmd = m.update(_hand_scaled(0.5, 0.5, palm_len=0.30), "Pointing_Up", 0.9)
    assert cmd.action == ACTION_FLY
    assert cmd.pitch > 5.0


def test_fly_mode_pitch_back_when_hand_far():
    cfg = MapperConfig(deadzone=0.05, fly_hold_frames=1, max_speed=50, expo=0.0,
                       pitch_baseline=0.18, pitch_deadband=0.1, pitch_gain=4.0)
    m = GestureMapper(config=cfg)
    # palm_len = 0.08 (well below baseline) → hand far from camera → backward
    cmd = m.update(_hand_scaled(0.5, 0.5, palm_len=0.08), "Pointing_Up", 0.9)
    assert cmd.action == ACTION_FLY
    assert cmd.pitch < -5.0


def test_one_euro_filter_converges_on_constant_signal():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t = 0.0
    last = None
    for _ in range(100):
        t += 1 / 30
        last = f(10.0, t)
    assert abs(last - 10.0) < 0.5


def test_scalar_ema_tracks_value():
    e = ScalarEMA(alpha=0.5)
    e(0.0)
    e(10.0)
    # After two updates with alpha=0.5: 0 → 5
    assert abs(e.value - 5.0) < 1e-6
