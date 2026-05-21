"""Entry point: webcam → MediaPipe Hands → gesture mapper → Codrone."""
from __future__ import annotations

import argparse
import sys

import cv2
import mediapipe as mp

from .drone_controller import DroneController
from .gesture_mapper import (
    ACTION_NONE,
    Command,
    GestureMapper,
    classify_static_gesture,
)
from .hand_tracker import HandTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-gesture control for Codrone EDU.")
    p.add_argument("--no-drone", action="store_true",
                   help="Dry-run: don't connect to the drone, just print commands.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    return p.parse_args()


def draw_overlay(frame, hand_landmarks, gesture: str, cmd: Command) -> None:
    h, w = frame.shape[:2]
    if hand_landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
        )
    text = f"gesture: {gesture}  action: {cmd.action}"
    if cmd.action == "fly":
        text += f"  P={cmd.pitch:+.0f} R={cmd.roll:+.0f} T={cmd.throttle:+.0f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "press q to land+quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    tracker = HandTracker(max_hands=1)
    mapper = GestureMapper()
    controller = DroneController(dry_run=args.no_drone)
    controller.connect()

    # Local MediaPipe Hands instance for drawing overlay only (tracker holds its own).
    mp_hands = mp.solutions.hands

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)  # selfie-view

            hands = tracker.process(frame)
            landmarks = hands[0].landmarks if hands else None
            controller.note_hand_seen(hands != [])

            cmd = mapper.update(landmarks)
            controller.apply(cmd)

            gesture = classify_static_gesture(landmarks) if landmarks is not None else "none"

            # Build a MediaPipe NormalizedLandmarkList just for drawing.
            hand_lm_proto = None
            if hands:
                from mediapipe.framework.formats import landmark_pb2
                hand_lm_proto = landmark_pb2.NormalizedLandmarkList()
                for x, y, z in hands[0].landmarks:
                    lm = hand_lm_proto.landmark.add()
                    lm.x, lm.y, lm.z = float(x), float(y), float(z)

            draw_overlay(frame, hand_lm_proto, gesture, cmd)
            cv2.imshow("Codrone Hand Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        controller.shutdown()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
