"""Hand tracking + gesture recognition using MediaPipe Tasks.

Uses Google's pretrained `gesture_recognizer.task` model, which gives both
21 hand landmarks AND a recognized gesture label out of:
    None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down,
    Thumb_Up, Victory, ILoveYou

The model file is auto-downloaded from MediaPipe's official model zoo on
first run, cached under ``models/``.
"""
from __future__ import annotations

import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Official MediaPipe-hosted model. Float16 keeps the file small (~8 MB) and runs
# on CPU. If you want max accuracy, swap to the float32 URL.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 3), normalized x/y in [0,1], z relative
    world_landmarks: Optional[np.ndarray]  # (21, 3) in meters, wrist-centered
    handedness: str        # "Left" / "Right" as labeled by the model (mirrored vs. selfie view)
    gesture: str           # one of model categories or "" / "None"
    gesture_score: float


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the gesture-recognizer .task file if it isn't on disk yet."""
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 1024:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading gesture model from %s", MODEL_URL)
    tmp = model_path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(model_path)
    logger.info("Model cached at %s (%d bytes)", model_path, model_path.stat().st_size)
    return model_path


class HandTracker:
    """Live-mode wrapper around MediaPipe Tasks GestureRecognizer."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_hands: int = 1,
        min_hand_detection_confidence: float = 0.6,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        path = ensure_model(model_path or DEFAULT_MODEL_PATH)
        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)
        self._mp_image_cls = self._import_image_cls()

    @staticmethod
    def _import_image_cls():
        import mediapipe as mp
        return mp.Image, mp.ImageFormat

    def process(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> List[Hand]:
        Image, ImageFormat = self._mp_image_cls
        # MediaPipe wants SRGB.
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        hands: List[Hand] = []
        if not result.hand_landmarks:
            return hands

        for i, lm_list in enumerate(result.hand_landmarks):
            arr = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
            world = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = np.array(
                    [[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]],
                    dtype=np.float32,
                )
            handedness = "Unknown"
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                handedness = result.handedness[i][0].category_name
            gesture = ""
            score = 0.0
            if result.gestures and i < len(result.gestures) and result.gestures[i]:
                gesture = result.gestures[i][0].category_name
                score = float(result.gestures[i][0].score)
            hands.append(Hand(
                landmarks=arr,
                world_landmarks=world,
                handedness=handedness,
                gesture=gesture,
                gesture_score=score,
            ))
        return hands

    def close(self) -> None:
        try:
            self._recognizer.close()
        except Exception:  # noqa: BLE001
            pass
