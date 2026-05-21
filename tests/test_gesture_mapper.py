"""Unit tests for the two-handed RC-stick gesture mapper."""
from __future__ import annotations

import numpy as np

from src.gesture_mapper import (
    ACTION_ESTOP,
    ACTION_FLY,
    ACTION_LAND,
    ACTION_NONE,
    ACTION_TAKEOFF,
    GestureMapper,
    HandObservation,
    MapperConfig,
)
from src.smoothing import OneEuroFilter, ScalarEMA


def _hand(wx: float, wy: float, gesture: str = "", score: float = 0.0) -> HandObservation:
    """Build a HandObservation whose palm center sits at (wx, wy)."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Place wrist + 4 MCPs all at the same (wx, wy) so palm center == (wx, wy).
    for i in (0, 5, 9, 13, 17):
        lm[i] = (wx, wy, 0.0)
    # Other landmarks don't matter for the two-handed mapper.
    return HandObservation(landmarks=lm, gesture=gesture, score=score)


def _two_hands(left_xy, right_xy, gesture="", score=0.0):
    return [
        _hand(*left_xy, gesture=gesture, score=score),
        _hand(*right_xy, gesture=gesture, score=score),
    ]


# --- empty / single-hand ---------------------------------------------------

def test_no_hands_returns_none():
    assert GestureMapper().update([]).action == ACTION_NONE


def test_single_hand_without_action_hovers():
    cmd = GestureMapper().update([_hand(0.25, 0.5)])
    assert cmd.action == ACTION_NONE


# --- action gestures -------------------------------------------------------

def test_two_open_palms_trigger_takeoff_after_hold():
    cfg = MapperConfig(hold_frames=3)
    m = GestureMapper(config=cfg)
    obs = _two_hands((0.25, 0.5), (0.75, 0.5), gesture="Open_Palm", score=0.9)
    for _ in range(cfg.hold_frames - 1):
        assert m.update(obs).action == ACTION_NONE
    assert m.update(obs).action == ACTION_TAKEOFF
    # Won't fire again until the gesture breaks.
    assert m.update(obs).action == ACTION_NONE


def test_one_open_palm_does_not_takeoff():
    cfg = MapperConfig(hold_frames=2)
    m = GestureMapper(config=cfg)
    obs = [_hand(0.25, 0.5, "Open_Palm", 0.9), _hand(0.75, 0.5, "Closed_Fist", 0.9)]
    for _ in range(20):
        assert m.update(obs).action != ACTION_TAKEOFF


def test_two_fists_trigger_land():
    cfg = MapperConfig(hold_frames=2)
    m = GestureMapper(config=cfg)
    obs = _two_hands((0.25, 0.5), (0.75, 0.5), gesture="Closed_Fist", score=0.9)
    m.update(obs)
    assert m.update(obs).action == ACTION_LAND


def test_single_thumb_down_triggers_estop():
    cfg = MapperConfig(estop_hold_frames=3)
    m = GestureMapper(config=cfg)
    obs = [_hand(0.25, 0.5, "Thumb_Down", 0.9)]
    for _ in range(cfg.estop_hold_frames - 1):
        assert m.update(obs).action == ACTION_NONE
    assert m.update(obs).action == ACTION_ESTOP


def test_low_score_does_not_trigger_actions():
    cfg = MapperConfig(hold_frames=2, min_gesture_score=0.7)
    m = GestureMapper(config=cfg)
    obs = _two_hands((0.25, 0.5), (0.75, 0.5), gesture="Open_Palm", score=0.3)
    for _ in range(20):
        assert m.update(obs).action != ACTION_TAKEOFF


# --- fly mode --------------------------------------------------------------

def _fly_cfg():
    return MapperConfig(deadzone=0.05, max_speed=50, expo=0.0)


def test_two_hands_at_neutrals_hover():
    cfg = _fly_cfg()
    cmd = GestureMapper(config=cfg).update(_two_hands(
        (cfg.left_neutral_x, cfg.left_neutral_y),
        (cfg.right_neutral_x, cfg.right_neutral_y),
    ))
    assert cmd.action == ACTION_FLY
    assert abs(cmd.pitch) < 0.5
    assert abs(cmd.roll) < 0.5
    assert abs(cmd.throttle) < 0.5
    assert abs(cmd.yaw) < 0.5


def test_right_hand_right_only_changes_roll():
    """Moving the right hand right should change roll, NOT pitch/throttle/yaw."""
    cfg = _fly_cfg()
    obs = _two_hands(
        (cfg.left_neutral_x, cfg.left_neutral_y),
        (cfg.right_neutral_x + cfg.stick_range, cfg.right_neutral_y),
    )
    cmd = GestureMapper(config=cfg).update(obs)
    assert cmd.action == ACTION_FLY
    assert cmd.roll > 5.0          # moved right → roll positive
    assert abs(cmd.pitch) < 0.5    # decoupled!
    assert abs(cmd.throttle) < 0.5 # decoupled!
    assert abs(cmd.yaw) < 0.5      # decoupled!


def test_right_hand_up_only_changes_pitch():
    cfg = _fly_cfg()
    obs = _two_hands(
        (cfg.left_neutral_x, cfg.left_neutral_y),
        (cfg.right_neutral_x, cfg.right_neutral_y - cfg.stick_range),
    )
    cmd = GestureMapper(config=cfg).update(obs)
    assert cmd.action == ACTION_FLY
    assert cmd.pitch > 5.0
    assert abs(cmd.roll) < 0.5
    assert abs(cmd.throttle) < 0.5
    assert abs(cmd.yaw) < 0.5


def test_left_hand_up_only_changes_throttle():
    cfg = _fly_cfg()
    obs = _two_hands(
        (cfg.left_neutral_x, cfg.left_neutral_y - cfg.stick_range),
        (cfg.right_neutral_x, cfg.right_neutral_y),
    )
    cmd = GestureMapper(config=cfg).update(obs)
    assert cmd.action == ACTION_FLY
    assert cmd.throttle > 5.0
    assert abs(cmd.pitch) < 0.5
    assert abs(cmd.roll) < 0.5
    assert abs(cmd.yaw) < 0.5


def test_left_hand_right_only_changes_yaw():
    cfg = _fly_cfg()
    obs = _two_hands(
        (cfg.left_neutral_x + cfg.stick_range, cfg.left_neutral_y),
        (cfg.right_neutral_x, cfg.right_neutral_y),
    )
    cmd = GestureMapper(config=cfg).update(obs)
    assert cmd.action == ACTION_FLY
    assert cmd.yaw > 5.0
    assert abs(cmd.pitch) < 0.5
    assert abs(cmd.roll) < 0.5
    assert abs(cmd.throttle) < 0.5


def test_hand_assignment_robust_to_input_order():
    """Sorting by X position means input order to update() shouldn't matter."""
    cfg = _fly_cfg()
    a = _hand(cfg.left_neutral_x + cfg.stick_range, cfg.left_neutral_y)  # left stick, right-of-neutral
    b = _hand(cfg.right_neutral_x, cfg.right_neutral_y)                  # right stick, neutral
    forward = GestureMapper(config=cfg).update([a, b])
    backward = GestureMapper(config=cfg).update([b, a])
    assert abs(forward.yaw - backward.yaw) < 1e-3
    assert abs(forward.roll - backward.roll) < 1e-3


# --- smoothing utilities ---------------------------------------------------

def test_one_euro_converges_on_constant():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t = 0.0
    out = None
    for _ in range(100):
        t += 1 / 30
        out = f(10.0, t)
    assert abs(out - 10.0) < 0.5


def test_scalar_ema_tracks_value():
    e = ScalarEMA(alpha=0.5)
    e(0.0); e(10.0)
    assert abs(e.value - 5.0) < 1e-6
