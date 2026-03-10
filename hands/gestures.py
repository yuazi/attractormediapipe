from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import (
    LUMINOSITY_RANGE,
    INDEX_Y_RANGE,
    PINCH_RANGE,
    PITCH_RANGE,
    SCALE_RANGE,
    SCENE_TURN_COOLDOWN_SECONDS,
    SCENE_SWITCH_PINKY_TOUCH_MAX_DISTANCE,
    SPEED_RANGE,
    YAW_RANGE,
)


Landmark = Tuple[float, float, float]
HandData = Dict[str, Optional[List[Landmark]]]


@dataclass
class GestureFrame:
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    scale: Optional[float] = None
    speed: Optional[float] = None
    luminosity: Optional[float] = None
    trail_len: Optional[int] = None
    scene_delta: int = 0
    left_detected: bool = False
    right_detected: bool = False


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def remap(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if math.isclose(src_min, src_max):
        return dst_min
    ratio = clamp((value - src_min) / (src_max - src_min), 0.0, 1.0)
    return dst_min + ratio * (dst_max - dst_min)


def wrist_position(landmarks: List[Landmark]) -> Tuple[float, float]:
    wrist = landmarks[0]
    return wrist[0], wrist[1]


def pinch_distance(landmarks: List[Landmark]) -> float:
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])


def palm_center(landmarks: List[Landmark]) -> Tuple[float, float]:
    palm_indices = (0, 5, 9, 13, 17)
    x = sum(landmarks[idx][0] for idx in palm_indices) / len(palm_indices)
    y = sum(landmarks[idx][1] for idx in palm_indices) / len(palm_indices)
    return x, y


def pinky_touch_distance(landmarks: List[Landmark]) -> float:
    center_x, center_y = palm_center(landmarks)
    pinky_tip = landmarks[20]
    return math.hypot(pinky_tip[0] - center_x, pinky_tip[1] - center_y)


class GestureInterpreter:
    def __init__(self) -> None:
        self._last_scene_turn_time = float("-inf")
        self._scene_turn_armed = {"left": True, "right": True}

    def update(self, hand_data: HandData, now: Optional[float] = None) -> GestureFrame:
        timestamp = time.monotonic() if now is None else now
        left = hand_data.get("left")
        right = hand_data.get("right")

        frame = GestureFrame(
            left_detected=left is not None,
            right_detected=right is not None,
        )

        if left:
            wrist_x, wrist_y = wrist_position(left)
            frame.yaw = remap(wrist_x, 0.0, 1.0, *YAW_RANGE)
            frame.pitch = remap(wrist_y, 0.0, 1.0, *PITCH_RANGE)
            frame.speed = remap(pinch_distance(left), *PINCH_RANGE, *SPEED_RANGE)
            frame.scene_delta += self._update_scene_turn("left", left, timestamp, -1)
        else:
            self._scene_turn_armed["left"] = True

        if right:
            frame.luminosity = remap(right[8][1], *INDEX_Y_RANGE, *LUMINOSITY_RANGE)
            frame.scale = remap(pinch_distance(right), *PINCH_RANGE, *SCALE_RANGE)
            frame.scene_delta += self._update_scene_turn("right", right, timestamp, 1)
        else:
            self._scene_turn_armed["right"] = True

        return frame

    def _update_scene_turn(self, hand_label: str, landmarks: List[Landmark], now: float, delta: int) -> int:
        if pinky_touch_distance(landmarks) > SCENE_SWITCH_PINKY_TOUCH_MAX_DISTANCE:
            self._scene_turn_armed[hand_label] = True
            return 0

        if not self._scene_turn_armed[hand_label] or now - self._last_scene_turn_time < SCENE_TURN_COOLDOWN_SECONDS:
            return 0

        self._scene_turn_armed[hand_label] = False
        self._last_scene_turn_time = now
        return delta
