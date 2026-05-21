"""Entry point: webcam → GestureRecognizer → mapper → Codrone."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .drone_controller import DroneController
from .gesture_mapper import ACTION_FLY, Command, GestureMapper, MapperConfig
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
    p = argparse.ArgumentParser(description="Hand-gesture control for Codrone EDU.")
    p.add_argument("--no-drone", action="store_true",
                   help="Dry-run: don't connect to the drone, just print commands.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    p.add_argument("--width", type=int, default=960, help="Capture width.")
    p.add_argument("--height", type=int, default=540, help="Capture height.")
    p.add_argument("--max-speed", type=int, default=50, help="Max stick magnitude (0-100).")
    p.add_argument("--deadzone", type=float, default=0.10, help="Deadzone fraction (0-1).")
    p.add_argument("--log", default="INFO", help="Log level (DEBUG/INFO/WARNING).")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def draw_landmarks(frame, landmarks_norm) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks_norm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)


def draw_hud(frame, *, fps: float, gesture: str, score: float,
             cmd: Command, airborne: bool, battery, dry_run: bool) -> None:
    h, w = frame.shape[:2]
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 64), (0, 0, 0), -1)
    state = "AIRBORNE" if airborne else "GROUNDED"
    color = (0, 200, 255) if airborne else (200, 200, 200)
    line1 = f"{state}  |  FPS {fps:5.1f}  |  gesture: {gesture or '-'} ({score:.2f})"
    if dry_run:
        line1 = "[DRY-RUN]  " + line1
    if battery is not None:
        line1 += f"  |  bat {battery}%"
    cv2.putText(frame, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    line2 = f"action: {cmd.action}"
    if cmd.action == ACTION_FLY:
        line2 += (
            f"   Fwd/Back {cmd.pitch:+5.0f}"
            f"   Left/Right {cmd.roll:+5.0f}"
            f"   Up/Down {cmd.throttle:+5.0f}"
            f"   Turn {cmd.yaw:+5.0f}"
        )
    cv2.putText(frame, line2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom hint
    cv2.putText(frame, "open palm = takeoff   fist = land   thumb down = E-stop   point up = fly   q = land+quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    # Center crosshair for fly-mode framing
    cx, cy = w // 2, h // 2
    cv2.drawMarker(frame, (cx, cy), (80, 80, 80), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    setup_logging(args.log)
    log = logging.getLogger("codrone_ai")

    # Pre-fetch the gesture model so the first frame isn't a 30-second stall.
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

    tracker = HandTracker(max_hands=1)
    mapper = GestureMapper(config=MapperConfig(
        max_speed=args.max_speed,
        deadzone=args.deadzone,
    ))
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
            frame = cv2.flip(frame, 1)  # selfie view

            ts_ms = int((time.monotonic() - start) * 1000)
            hands = tracker.process(frame, timestamp_ms=ts_ms)
            controller.note_hand_seen(bool(hands))

            landmarks = hands[0].landmarks if hands else None
            gesture = hands[0].gesture if hands else ""
            score = hands[0].gesture_score if hands else 0.0

            cmd = mapper.update(landmarks, gesture=gesture, score=score)
            controller.apply(cmd)

            now = time.monotonic()
            dt = max(now - last_t, 1e-6)
            last_t = now
            fps = fps_ema(1.0 / dt)

            if landmarks is not None:
                draw_landmarks(frame, landmarks)
            draw_hud(
                frame, fps=fps, gesture=gesture, score=score, cmd=cmd,
                airborne=controller.airborne, battery=controller.battery(),
                dry_run=args.no_drone,
            )
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
