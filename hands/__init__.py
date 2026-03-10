from .gestures import GestureFrame, GestureInterpreter

__all__ = [
    "GestureFrame",
    "GestureInterpreter",
    "HAND_CONNECTIONS",
    "MEDIAPIPE_AVAILABLE",
    "HandTracker",
    "draw_hand_skeleton",
]


def __getattr__(name):
    if name in {"HAND_CONNECTIONS", "draw_hand_skeleton"}:
        from .skeleton import HAND_CONNECTIONS, draw_hand_skeleton

        return HAND_CONNECTIONS if name == "HAND_CONNECTIONS" else draw_hand_skeleton
    if name in {"MEDIAPIPE_AVAILABLE", "HandTracker"}:
        from .tracker import MEDIAPIPE_AVAILABLE, HandTracker

        return MEDIAPIPE_AVAILABLE if name == "MEDIAPIPE_AVAILABLE" else HandTracker
    raise AttributeError(f"module 'hands' has no attribute {name!r}")
