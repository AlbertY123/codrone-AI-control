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
`models/gesture_recognizer.task`). Recognized labels drive these actions:

| Gesture           | Action                              |
|-------------------|-------------------------------------|
| Open Palm         | Takeoff (hold ~0.3 s)               |
| Closed Fist       | Land                                |
| Thumb Down        | Emergency stop (longer hold)        |
| Pointing Up       | Continuous flight mode              |
| (any other hand)  | Hover / no input                    |

In continuous flight mode:
- Palm X offset from frame center → **roll**
- Palm Y offset from frame center → **throttle**
- Wrist depth (palm closer/farther from camera) → **pitch**
- Hand roll angle (line from pinky to index MCP) → **yaw**

Outputs are deadzoned, expo-shaped, and One-Euro-filtered so the drone holds
position cleanly when your hand is near the center. Press **q** at any time to
land and quit.

### Production-grade behaviors

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
