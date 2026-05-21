# Codrone Hand-Gesture Control — Plan

## Goal
Control a Codrone EDU drone in real time using hand gestures captured by the desktop webcam.

## Stack
- **Drone SDK:** `codrone_edu` (same library used in scotej/Codrone-Testing/events.py — exposes `takeoff()`, `land()`, `emergency_stop()`, `set_pitch/roll/throttle/yaw(speed)`, `move()`).
- **Hand detection AI:** **MediaPipe Hands** (Google). Runs locally on CPU, returns 21 3D landmarks per hand at ~30 FPS, free, mature, well-documented. No cloud key needed.
- **Webcam I/O:** OpenCV (`opencv-python`).
- **Language:** Python 3.10–3.11 (MediaPipe wheels don't yet target 3.12+ reliably on Windows).

## Project layout
```
Codrone_AI_Control/
├── PLAN.md
├── requirements.txt
├── README.md
├── src/
│   ├── main.py              # entry point: wires camera → gesture → drone
│   ├── hand_tracker.py      # MediaPipe wrapper, returns landmarks + handedness
│   ├── gesture_mapper.py    # landmarks → high-level command (takeoff, land, pitch, roll, ...)
│   └── drone_controller.py  # thin codrone_edu wrapper with safety guards
└── tests/
    └── test_gesture_mapper.py  # unit tests on synthetic landmark inputs (no drone, no camera)
```

## Gesture → command mapping (initial set)

| Gesture (right hand)                     | Drone action                          |
|------------------------------------------|---------------------------------------|
| Open palm, all 5 fingers up              | **Takeoff** (one-shot, debounced)     |
| Closed fist                              | **Land** (one-shot, debounced)        |
| Thumbs up + others closed                | **Emergency stop**                    |
| Index finger only, palm-tilt steers      | **Continuous flight mode** (see below)|

In continuous flight mode, the wrist position inside the frame drives motion proportionally:
- Wrist X offset from frame center → **roll** (left/right)
- Wrist Y offset from frame center → **throttle** (up/down)
- Hand pitch (angle of index finger vs. wrist) → **pitch** (forward/back)
- Two-hand mode (optional, phase 2): left-hand X offset → **yaw**

Speed is clamped to ±50 (codrone_edu range −100..100) and zeroed inside a deadzone near frame center so the drone hovers cleanly.

## Safety guards (in `drone_controller.py`)
- Loud `emergency_stop()` bound to **`q` key** + thumbs-up gesture + loss-of-hand-for-N-frames → auto `land()`.
- One-shot gestures (takeoff/land) require the gesture to hold for ~0.5 s (≈15 frames at 30 FPS) to avoid spurious triggers.
- Hard cap on `set_*` magnitudes regardless of mapping output.
- `--no-drone` CLI flag: prints commands instead of sending them, for camera/gesture testing without hardware.

## Implementation steps
1. **Scaffold + deps.** Create `requirements.txt` (`codrone-edu`, `mediapipe`, `opencv-python`, `numpy`), `README.md` with run instructions, `.gitignore`.
2. **`hand_tracker.py`.** Wrap `mp.solutions.hands.Hands`; expose `process(frame) -> list[Hand]` with landmarks normalized to [0,1] plus handedness label.
3. **`gesture_mapper.py`.** Pure function from landmarks → `Command` dataclass `(action: str, pitch, roll, throttle, yaw)`. Includes finger-up detection (compare tip vs. PIP joint Y, with thumb special-cased on X). Has the debounce state machine for one-shot gestures.
4. **`drone_controller.py`.** Wraps `Drone` from `codrone_edu`. Methods: `connect()`, `apply(Command)`, `shutdown()`. Implements speed clamp, hand-lost timeout, and `--no-drone` dry-run mode.
5. **`main.py`.** Loop: read webcam → tracker → mapper → controller → overlay landmarks + current command on the frame → `cv2.imshow`. `q` to quit (triggers `land`).
6. **Unit tests** in `tests/test_gesture_mapper.py` for the finger-up logic and debounce, using hand-crafted fake landmark arrays. No drone, no camera needed.
7. **Dry run** with `--no-drone` to validate gesture detection on-screen.
8. **Hardware bring-up.** Pair drone over USB dongle, run without flags in an open space, takeoff/land first, then continuous mode.

## Open question worth flagging before coding
The codrone_edu API uses `set_*` + a single `move()` call that flies for a duration. For a real-time loop we'll call `set_*` every frame and `move(0.05)` (50 ms) per iteration so motion tracks the hand. If latency is poor on your machine we can fall back to `send_control()` directly. I'll prototype with `move()` first.

## Out of scope (phase 2 ideas)
- Two-handed yaw control.
- Voice command fallback.
- Recording flight telemetry to CSV.
