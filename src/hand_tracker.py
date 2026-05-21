"""MediaPipe Hands wrapper.

Returns a list of detected hands per frame, each with 21 normalized landmarks
and a handedness label ("Left"/"Right" from the model's point of view, which
is mirrored vs. a selfie-view webcam — see notes in gesture_mapper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import mediapipe as mp
import numpy as np


@dataclass
class Hand:
    landmarks: np.ndarray  # shape (21, 3), values in [0, 1] for x/y, relative z
    handedness: str        # "Left" or "Right" as labeled by MediaPipe
    score: float


class HandTracker:
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[Hand]:
        # MediaPipe expects RGB.
        frame_rgb = frame_bgr[:, :, ::-1]
        result = self._hands.process(frame_rgb)
        hands: List[Hand] = []
        if not result.multi_hand_landmarks:
            return hands
        for lm, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            label = hd.classification[0].label
            score = float(hd.classification[0].score)
            hands.append(Hand(landmarks=arr, handedness=label, score=score))
        return hands

    def close(self) -> None:
        self._hands.close()
