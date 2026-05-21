"""Thin wrapper around codrone_edu with safety guards.

- Speed clamp on every set_* call.
- Hand-lost timer auto-lands if no hand is seen for `hand_lost_seconds`.
- --no-drone dry-run prints commands instead of sending them.
"""
from __future__ import annotations

import time
from typing import Optional

from .gesture_mapper import (
    ACTION_ESTOP,
    ACTION_FLY,
    ACTION_LAND,
    ACTION_NONE,
    ACTION_TAKEOFF,
    Command,
)


SPEED_CAP = 60  # absolute max applied regardless of mapper output


class DroneController:
    def __init__(self, dry_run: bool = False, hand_lost_seconds: float = 1.5):
        self.dry_run = dry_run
        self.hand_lost_seconds = hand_lost_seconds
        self._drone = None
        self._airborne = False
        self._last_hand_seen = time.monotonic()

    def connect(self) -> None:
        if self.dry_run:
            print("[dry-run] not connecting to drone")
            return
        from codrone_edu.drone import Drone  # imported lazily so dry-run needs no SDK
        self._drone = Drone()
        self._drone.pair()
        print("[drone] paired")

    def shutdown(self) -> None:
        if self.dry_run or self._drone is None:
            return
        try:
            if self._airborne:
                self._drone.land()
        finally:
            self._drone.close()

    def note_hand_seen(self, seen: bool) -> None:
        if seen:
            self._last_hand_seen = time.monotonic()

    def _hand_lost(self) -> bool:
        return (time.monotonic() - self._last_hand_seen) > self.hand_lost_seconds

    @staticmethod
    def _clamp(v: float) -> int:
        return int(max(-SPEED_CAP, min(SPEED_CAP, v)))

    def apply(self, cmd: Command) -> None:
        # Safety: if we've lost the hand and we're airborne, land.
        if self._airborne and self._hand_lost():
            self._do_land(reason="hand lost")
            return

        if cmd.action == ACTION_TAKEOFF and not self._airborne:
            self._do(lambda: self._drone.takeoff(), "takeoff")
            self._airborne = True
        elif cmd.action == ACTION_LAND and self._airborne:
            self._do_land(reason="gesture")
        elif cmd.action == ACTION_ESTOP:
            self._do(lambda: self._drone.emergency_stop(), "emergency_stop")
            self._airborne = False
        elif cmd.action == ACTION_FLY and self._airborne:
            self._set_motion(cmd)
        elif cmd.action == ACTION_NONE and self._airborne:
            # Hover: zero stick inputs.
            self._set_motion(Command(action=ACTION_FLY))

    def _do_land(self, reason: str) -> None:
        self._do(lambda: self._drone.land(), f"land ({reason})")
        self._airborne = False

    def _set_motion(self, cmd: Command) -> None:
        p = self._clamp(cmd.pitch)
        r = self._clamp(cmd.roll)
        t = self._clamp(cmd.throttle)
        y = self._clamp(cmd.yaw)
        if self.dry_run:
            print(f"[dry-run] pitch={p} roll={r} throttle={t} yaw={y}")
            return
        self._drone.set_pitch(p)
        self._drone.set_roll(r)
        self._drone.set_throttle(t)
        self._drone.set_yaw(y)
        # Short move slice so the loop stays responsive (~20 Hz).
        self._drone.move(0.05)

    def _do(self, fn, label: str) -> None:
        if self.dry_run:
            print(f"[dry-run] {label}")
            return
        print(f"[drone] {label}")
        fn()
