from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from config import HAND_TRACKING_FPS, TRACKING_FRAME_HEIGHT, TRACKING_FRAME_WIDTH

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker,
        HandLandmarkerOptions,
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )

    MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
    mp = None
    MEDIAPIPE_AVAILABLE = False

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "hand_landmarker.task"
)


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        raise RuntimeError(f"Missing bundled MediaPipe model: {_MODEL_PATH}")
    return _MODEL_PATH


Landmark = Tuple[float, float, float]
HandMap = Dict[str, Optional[List[Landmark]]]


def _normalize_handedness_label(label: str) -> Optional[str]:
    normalized = label.strip().lower()
    if normalized == "left":
        return "right"
    if normalized == "right":
        return "left"
    return None


def _hand_center_x(landmarks: Sequence[Landmark]) -> float:
    palm_indices = (0, 5, 9, 13, 17)
    return sum(landmarks[idx][0] for idx in palm_indices) / len(palm_indices)


def _assign_hand_slots(
    detections: Sequence[Tuple[Optional[str], List[Landmark]]],
) -> HandMap:
    output: HandMap = {"left": None, "right": None}
    if not detections:
        return output

    if len(detections) == 1:
        label, landmarks = detections[0]
        if label is None:
            label = "left" if _hand_center_x(landmarks) < 0.5 else "right"
        output[label] = landmarks
        return output

    labeled: dict[str, list[List[Landmark]]] = {"left": [], "right": []}
    unlabeled: list[List[Landmark]] = []
    for label, landmarks in detections:
        if label in labeled:
            labeled[label].append(landmarks)
        else:
            unlabeled.append(landmarks)

    # Trust MediaPipe's handedness when each slot has a single clear match.
    for slot in ("left", "right"):
        if len(labeled[slot]) == 1:
            output[slot] = labeled[slot][0]

    pending: list[List[Landmark]] = []
    for slot in ("left", "right"):
        if len(labeled[slot]) != 1:
            pending.extend(labeled[slot])
    pending.extend(unlabeled)

    if not pending:
        return output

    ranked = sorted(pending, key=_hand_center_x)
    if output["left"] is None and output["right"] is None:
        output["left"] = ranked[0]
        output["right"] = ranked[-1]
        return output

    if output["left"] is None:
        output["left"] = ranked[0]
    if output["right"] is None:
        output["right"] = ranked[-1]
    return output


class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        confidence: float = 0.7,
        max_fps: int = HAND_TRACKING_FPS,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe and opencv-python are required for hand tracking")
        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_time_ms = int(time.perf_counter() * 1000)
        self._last_process_ms = -10**9
        self._last_timestamp_ms = -1
        self._min_frame_interval_ms = 1000.0 / max(1, max_fps)
        self._last_output: HandMap = {"left": None, "right": None}

    def process(self, frame_bgr) -> HandMap:
        if not MEDIAPIPE_AVAILABLE:
            return {"left": None, "right": None}

        now_ms = int(time.perf_counter() * 1000)
        if now_ms - self._last_process_ms < self._min_frame_interval_ms:
            return self._last_output

        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w != TRACKING_FRAME_WIDTH or frame_h != TRACKING_FRAME_HEIGHT:
            frame_bgr = cv2.resize(
                frame_bgr,
                (TRACKING_FRAME_WIDTH, TRACKING_FRAME_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(now_ms - self._start_time_ms, self._last_timestamp_ms + 1)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        self._last_process_ms = now_ms
        self._last_timestamp_ms = timestamp_ms

        if not result.hand_landmarks or not result.handedness:
            self._last_output = {"left": None, "right": None}
            return self._last_output

        detections: list[tuple[Optional[str], List[Landmark]]] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # The camera feed is mirrored before tracking. MediaPipe's handedness
            # labels follow that mirrored image space here, so normalize them
            # back to the user's physical left/right hands for gesture mapping.
            label = _normalize_handedness_label(handedness[0].category_name)
            if label is None:
                label = None
            points = [(lm.x, lm.y, lm.z) for lm in landmarks]
            detections.append((label, points))

        output = _assign_hand_slots(detections)
        self._last_output = output
        return output

    def close(self) -> None:
        if MEDIAPIPE_AVAILABLE:
            self._landmarker.close()
