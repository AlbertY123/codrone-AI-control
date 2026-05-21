"""codrone_edu wrapper with safety guards.

- Speed clamp on every set_* call.
- Hand-lost watchdog: auto-land after N seconds without a hand.
- One-shot actions are idempotent (won't double-fire takeoff/land).
- Pair retries with exponential backoff.
- --no-drone dry-run prints commands; useful for camera/gesture testing.
- Optional battery polling (silently no-ops if the SDK doesn't expose it).
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

SPEED_CAP = 70  # hard cap regardless of mapper output


class DroneController:
    def __init__(
        self,
        dry_run: bool = False,
        hand_lost_seconds: float = 1.2,
        pair_retries: int = 3,
        move_slice_seconds: float = 0.05,
    ):
        self.dry_run = dry_run
        self.hand_lost_seconds = hand_lost_seconds
        self.pair_retries = pair_retries
        self.move_slice_seconds = move_slice_seconds
        self._drone = None
        self._airborne = False
        self._last_hand_seen = time.monotonic()
        self._last_battery_poll = 0.0
        self._battery_pct: Optional[int] = None

    # --- lifecycle -------------------------------------------------------------

    def connect(self) -> None:
        if self.dry_run:
            logger.info("dry-run: not connecting to drone")
            return
        from codrone_edu.drone import Drone  # lazy
        self._drone = Drone()
        for attempt in range(1, self.pair_retries + 1):
            try:
                self._drone.pair()
                logger.info("drone paired (attempt %d)", attempt)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("pair attempt %d failed: %s", attempt, e)
                time.sleep(min(2.0 * attempt, 5.0))
        raise RuntimeError("failed to pair Codrone after retries; check USB dongle and battery")

    def shutdown(self) -> None:
        if self.dry_run or self._drone is None:
            return
        try:
            if self._airborne:
                logger.info("shutdown: landing")
                self._safe_call(self._drone.land)
        finally:
            self._safe_call(self._drone.close)

    # --- input -----------------------------------------------------------------

    def note_hand_seen(self, seen: bool) -> None:
        if seen:
            self._last_hand_seen = time.monotonic()

    def _hand_lost(self) -> bool:
        return (time.monotonic() - self._last_hand_seen) > self.hand_lost_seconds

    def battery(self) -> Optional[int]:
        """Best-effort battery percentage (1 Hz polling)."""
        if self.dry_run or self._drone is None:
            return None
        now = time.monotonic()
        if now - self._last_battery_poll < 1.0:
            return self._battery_pct
        self._last_battery_poll = now
        try:
            self._battery_pct = int(self._drone.get_battery())
        except Exception:  # noqa: BLE001
            self._battery_pct = None
        return self._battery_pct

    # --- core ------------------------------------------------------------------

    def apply(self, cmd: Command) -> None:
        # Watchdog: auto-land if airborne and we've lost the hand.
        if self._airborne and self._hand_lost():
            logger.warning("hand lost > %.1fs → auto-land", self.hand_lost_seconds)
            self._do_land(reason="hand lost")
            return

        if cmd.action == ACTION_TAKEOFF:
            if not self._airborne:
                self._do(self._safe_takeoff, "takeoff")
                self._airborne = True
        elif cmd.action == ACTION_LAND:
            if self._airborne:
                self._do_land(reason="gesture")
        elif cmd.action == ACTION_ESTOP:
            self._do(lambda: self._safe_call(self._drone.emergency_stop), "emergency_stop")
            self._airborne = False
        elif cmd.action == ACTION_FLY and self._airborne:
            self._set_motion(cmd)
        elif cmd.action == ACTION_NONE and self._airborne:
            # No gesture: hover with zeroed sticks.
            self._set_motion(Command(action=ACTION_FLY))

    # --- helpers ---------------------------------------------------------------

    def _safe_takeoff(self) -> None:
        self._safe_call(self._drone.takeoff)

    def _do_land(self, reason: str) -> None:
        self._do(lambda: self._safe_call(self._drone.land), f"land ({reason})")
        self._airborne = False

    @staticmethod
    def _clamp(v: float) -> int:
        return int(max(-SPEED_CAP, min(SPEED_CAP, round(v))))

    def _set_motion(self, cmd: Command) -> None:
        p = self._clamp(cmd.pitch)
        r = self._clamp(cmd.roll)
        t = self._clamp(cmd.throttle)
        y = self._clamp(cmd.yaw)
        if self.dry_run:
            logger.debug("[dry-run] pitch=%d roll=%d throttle=%d yaw=%d", p, r, t, y)
            return
        d = self._drone
        self._safe_call(lambda: d.set_pitch(p))
        self._safe_call(lambda: d.set_roll(r))
        self._safe_call(lambda: d.set_throttle(t))
        self._safe_call(lambda: d.set_yaw(y))
        self._safe_call(lambda: d.move(self.move_slice_seconds))

    def _do(self, fn, label: str) -> None:
        if self.dry_run:
            logger.info("[dry-run] %s", label)
            return
        logger.info("drone: %s", label)
        fn()

    @staticmethod
    def _safe_call(fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.error("drone call failed: %s", e)

    # --- properties ------------------------------------------------------------

    @property
    def airborne(self) -> bool:
        return self._airborne
