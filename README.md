# Codrone AI Hand-Gesture Control

Control a Codrone EDU drone with hand gestures captured by your desktop webcam.
Hand detection runs locally with Google's MediaPipe Hands model; drone commands
go through the `codrone_edu` SDK (same library used in
[scotej/Codrone-Testing](https://github.com/scotej/Codrone-Testing)).

## Setup

Requires **Python 3.10 or 3.11**. MediaPipe does not yet ship wheels for
Python 3.12+, so newer versions will fail at `pip install mediapipe`.

### 1. Install Python 3.11 (if you don't have it)

Pick whichever is easiest for you — all three give you a separate 3.11 install
that lives alongside any newer Python you already have:

**Option A — winget (recommended, one line):**
```powershell
winget install --id Python.Python.3.11 -e
```

**Option B — official installer:**
Download from <https://www.python.org/downloads/release/python-3119/> and run
the installer. On the first screen check **"Add python.exe to PATH"**, then
finish with the default options.

**Option C — pyenv-win (best if you juggle multiple Python versions):**
```powershell
pip install pyenv-win --target $HOME\.pyenv
# add $HOME\.pyenv\pyenv-win\bin and \shims to PATH, then restart shell
pyenv install 3.11.9
pyenv local 3.11.9
```

Verify it's installed (don't worry if `python` still points at a newer one — we
target 3.11 explicitly in the next step):
```powershell
py -3.11 --version    # should print: Python 3.11.x
```

### 2. Create the venv with Python 3.11 and install deps

From the project root (`C:\Users\alber\Desktop\Web\Codrone_AI_Control`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version            # confirm: Python 3.11.x
pip install --upgrade pip
pip install -r requirements.txt
```

If `Activate.ps1` is blocked by execution policy, run this once per shell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If you installed via pyenv-win instead of `py`, replace `py -3.11` with the
full path, e.g. `$HOME\.pyenv\pyenv-win\versions\3.11.9\python.exe`.

Pair the Codrone USB dongle before running with hardware.

## Run

Dry-run (camera + gesture detection only, no drone):

```powershell
python -m src.main --no-drone
```

With drone:

```powershell
python -m src.main
```

## Gestures

Hand detection and gesture classification use Google's pretrained
**MediaPipe GestureRecognizer** model (auto-downloaded on first run to
`models/gesture_recognizer.task`). Recognized hand shapes do these things:

| Hand shape           | What the drone does                   |
|----------------------|---------------------------------------|
| Open palm            | Takes off (hold for ~0.3 s)           |
| Closed fist          | Lands                                 |
| Thumb down           | Emergency stop (hold longer)          |
| Index finger up      | Enters fly mode — see below           |
| (anything else)      | Hovers in place                       |

### Fly mode — how your hand moves the drone

Once you raise your index finger and the drone enters fly mode, move your hand
around in front of the camera:

| Hand motion                                  | Drone reaction          |
|----------------------------------------------|-------------------------|
| Move hand **left / right**                   | Drone slides left/right |
| Move hand **up / down**                      | Drone rises / lowers    |
| Move hand **closer to / farther from camera**| Drone flies forward / back |
| **Tilt hand** clockwise / counter-clockwise  | Drone turns right / left|

When your hand is near the middle of the frame at a normal distance, all four
controls sit at zero and the drone hovers. There's a small dead-zone around
the center so tiny hand wobbles don't move the drone, and outputs are smoothed
by a One-Euro filter so motion stays clean.

Press **q** at any time to land and quit.

### Under the hood

- Pretrained gesture model (not hand-rolled finger heuristics).
- One-Euro adaptive filter on stick outputs — smooth at rest, responsive in motion.
- One-shot gesture debouncing prevents accidental takeoff/land.
- Hand-lost watchdog: auto-land after ~1.2 s without a hand in frame.
- Hard speed cap on every output, applied below the mapper.
- Pair retries on connect; battery polling shown in HUD.
- DirectShow capture backend on Windows + low capture buffer for low latency.

## Tests

```powershell
python -m pytest tests/
```
