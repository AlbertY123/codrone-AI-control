"""Entry point: webcam → GestureRecognizer → two-handed mapper → Codrone."""
from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from .drone_controller import DroneController
from .gesture_mapper import (
    ACTION_FLY,
    Command,
    GestureMapper,
    HandObservation,
    MapperConfig,
)
from .hand_tracker import HandTracker, ensure_model
from .smoothing import ScalarEMA


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-handed hand-gesture control for Codrone EDU.")
    p.add_argument("--no-drone", action="store_true",
                   help="Dry-run: don't connect to the drone, just print commands.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--max-speed", type=int, default=50)
    p.add_argument("--log", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def draw_landmarks(frame, landmarks_norm, color=(0, 200, 0)) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks_norm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)


def draw_stick_zone(frame, cx_frac, cy_frac, range_frac, label_top, label_side,
                    hand_pos=None, active=False) -> None:
    h, w = frame.shape[:2]
    cx, cy = int(cx_frac * w), int(cy_frac * h)
    rx, ry = int(range_frac * w), int(range_frac * h)
    color = (0, 220, 255) if active else (110, 110, 110)
    # Outer box (stick range)
    cv2.rectangle(frame, (cx - rx, cy - ry), (cx + rx, cy + ry), color, 1, cv2.LINE_AA)
    # Center cross (neutral / hover)
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    # Axis labels (top of box, left of box)
    cv2.putText(frame, label_top, (cx - 40, cy - ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(frame, label_side, (cx - rx - 4, cy + ry + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    # Hand position marker
    if hand_pos is not None:
        hx, hy = int(hand_pos[0] * w), int(hand_pos[1] * h)
        cv2.circle(frame, (hx, hy), 8, (0, 255, 0) if active else (120, 120, 120),
                   2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (hx, hy),
                 (0, 255, 0) if active else (120, 120, 120), 1, cv2.LINE_AA)


def palm_center(lm: np.ndarray) -> np.ndarray:
    return lm[[0, 5, 9, 13, 17]].mean(axis=0)


def draw_hud(frame, *, fps, hands, cmd, controller, dry_run, cfg) -> None:
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    state = "AIRBORNE" if controller.airborne else "GROUNDED"
    color = (0, 200, 255) if controller.airborne else (200, 200, 200)
    n_hands = len(hands)
    head = f"{state}  |  FPS {fps:5.1f}  |  hands: {n_hands}/2  |  action: {cmd.action}"
    if dry_run:
        head = "[DRY-RUN]  " + head
    bat = controller.battery()
    if bat is not None:
        head += f"  |  bat {bat}%"
    cv2.putText(frame, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)

    if cmd.action == ACTION_FLY:
        sticks = (f"  Fwd/Back {cmd.pitch:+5.0f}   Left/Right {cmd.roll:+5.0f}"
                  f"   Up/Down {cmd.throttle:+5.0f}   Turn {cmd.yaw:+5.0f}")
        cv2.putText(frame, sticks, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Stick zones — both visible all the time so user knows the layout
    ordered = sorted(hands, key=lambda hh: float(hh.landmarks[0, 0]))
    left_hand = ordered[0] if len(ordered) >= 1 else None
    right_hand = ordered[-1] if len(ordered) >= 2 else None
    active = (len(ordered) >= 2 and cmd.action == ACTION_FLY)

    draw_stick_zone(
        frame, cfg.left_neutral_x, cfg.left_neutral_y, cfg.stick_range,
        "Up / Down", "Turn Left/Right",
        hand_pos=palm_center(left_hand.landmarks) if left_hand else None,
        active=active,
    )
    draw_stick_zone(
        frame, cfg.right_neutral_x, cfg.right_neutral_y, cfg.stick_range,
        "Fwd / Back", "Strafe Left/Right",
        hand_pos=palm_center(right_hand.landmarks) if right_hand else None,
        active=active,
    )

    # Footer
    cv2.putText(frame,
                "both palms = takeoff   both fists = land   thumb down = E-stop   q = land+quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (180, 180, 180), 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    setup_logging(args.log)
    log = logging.getLogger("codrone_ai")

    log.info("ensuring gesture model is cached…")
    ensure_model()

    log.info("opening camera %d", args.camera)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        log.error("could not open camera %d", args.camera)
        return 1

    tracker = HandTracker(max_hands=2)
    cfg = MapperConfig(max_speed=args.max_speed)
    mapper = GestureMapper(config=cfg)
    controller = DroneController(dry_run=args.no_drone)
    controller.connect()

    fps_ema = ScalarEMA(alpha=0.1)
    last_t = time.monotonic()
    start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("camera read failed; retrying")
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)

            ts_ms = int((time.monotonic() - start) * 1000)
            hands = tracker.process(frame, timestamp_ms=ts_ms)
            controller.note_hand_seen(bool(hands))

            observations = [
                HandObservation(landmarks=h.landmarks, gesture=h.gesture, score=h.gesture_score)
                for h in hands
            ]
            cmd = mapper.update(observations, timestamp=time.monotonic())
            controller.apply(cmd)

            now = time.monotonic()
            dt = max(now - last_t, 1e-6)
            last_t = now
            fps = fps_ema(1.0 / dt)

            for h in hands:
                draw_landmarks(frame, h.landmarks)

            draw_hud(frame, fps=fps, hands=observations, cmd=cmd,
                     controller=controller, dry_run=args.no_drone, cfg=cfg)
            cv2.imshow("Codrone Hand Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("q pressed → land+quit")
                break
    finally:
        controller.shutdown()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
