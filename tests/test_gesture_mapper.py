"""Unit tests for the pure gesture mapper.

We hand-build 21x3 landmark arrays that satisfy the geometric checks in
gesture_mapper.fingers_up, without any MediaPipe dependency.
"""
from __future__ import annotations

import numpy as np

from src.gesture_mapper import (
    ACTION_FLY,
    ACTION_LAND,
    ACTION_NONE,
    ACTION_TAKEOFF,
    GestureMapper,
    MapperConfig,
    classify_static_gesture,
    fingers_up,
)


def _base_hand() -> np.ndarray:
    """A neutral hand centered at (0.5, 0.5) with all fingers folded (fist)."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    lm[0] = (0.5, 0.5, 0.0)
    # Thumb chain (folded across the palm, tip near IP joint in x)
    lm[2] = (0.50, 0.50, 0.0)  # MCP
    lm[3] = (0.51, 0.50, 0.0)  # IP
    lm[4] = (0.51, 0.51, 0.0)  # tip: small x diff, y slightly below IP → not "up"
    # Other fingers folded: tips are below (greater y) than PIPs
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[pip] = (0.5, 0.45, 0.0)
        lm[tip] = (0.5, 0.50, 0.0)
    return lm


def _raise_finger(lm: np.ndarray, tip: int, pip: int) -> None:
    lm[pip] = (0.5, 0.45, 0.0)
    lm[tip] = (0.5, 0.35, 0.0)  # tip well above PIP


def _open_palm() -> np.ndarray:
    lm = _base_hand()
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        _raise_finger(lm, tip, pip)
    # Thumb up: sideways out
    lm[2] = (0.50, 0.50, 0.0)
    lm[4] = (0.60, 0.48, 0.0)  # large |dx|, y above IP
    return lm


def _thumbs_up() -> np.ndarray:
    lm = _base_hand()
    lm[2] = (0.50, 0.50, 0.0)
    lm[4] = (0.60, 0.40, 0.0)  # thumb up
    return lm


def _index_only(wrist_x=0.5, wrist_y=0.5) -> np.ndarray:
    lm = _base_hand()
    lm[0] = (wrist_x, wrist_y, 0.0)
    _raise_finger(lm, 8, 6)
    return lm


def test_fingers_up_fist():
    assert fingers_up(_base_hand()) == [False, False, False, False, False]


def test_classify_open_palm():
    assert classify_static_gesture(_open_palm()) == "open_palm"


def test_classify_fist():
    assert classify_static_gesture(_base_hand()) == "fist"


def test_classify_thumbs_up():
    assert classify_static_gesture(_thumbs_up()) == "thumbs_up"


def test_classify_index_only():
    assert classify_static_gesture(_index_only()) == "index_only"


def test_takeoff_requires_hold():
    cfg = MapperConfig(hold_frames=5)
    m = GestureMapper(config=cfg)
    palm = _open_palm()
    for _ in range(cfg.hold_frames - 1):
        assert m.update(palm).action == ACTION_NONE
    assert m.update(palm).action == ACTION_TAKEOFF
    # Subsequent frames of the same gesture do not refire.
    assert m.update(palm).action == ACTION_NONE


def test_land_after_takeoff():
    cfg = MapperConfig(hold_frames=2)
    m = GestureMapper(config=cfg)
    palm = _open_palm()
    fist = _base_hand()
    m.update(palm); m.update(palm)  # takeoff fires
    m.update(fist); cmd = m.update(fist)
    assert cmd.action == ACTION_LAND


def test_fly_deadzone_hovers():
    cfg = MapperConfig(deadzone=0.2, fly_hold_frames=1)
    m = GestureMapper(config=cfg)
    cmd = m.update(_index_only(0.5, 0.5))
    assert cmd.action == ACTION_FLY
    assert cmd.roll == 0.0 and cmd.throttle == 0.0


def test_fly_outside_deadzone_moves_right():
    cfg = MapperConfig(deadzone=0.1, fly_hold_frames=1, max_speed=50)
    m = GestureMapper(config=cfg)
    cmd = m.update(_index_only(0.9, 0.5))  # wrist far right
    assert cmd.action == ACTION_FLY
    assert cmd.roll > 0.0
    assert cmd.throttle == 0.0


def test_no_hand_returns_none_and_resets_streak():
    m = GestureMapper(config=MapperConfig(hold_frames=3))
    palm = _open_palm()
    m.update(palm); m.update(palm)  # streak=2
    assert m.update(None).action == ACTION_NONE
    # After hand loss, need a fresh full hold to fire again.
    assert m.update(palm).action == ACTION_NONE
    assert m.update(palm).action == ACTION_NONE
    assert m.update(palm).action == ACTION_TAKEOFF
