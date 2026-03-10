from __future__ import annotations

import math

import numpy as np

from config import CAMERA_DISTANCE


def perspective_matrix(fov_y_radians: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y_radians * 0.5)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def rotation_x(angle_radians: float) -> np.ndarray:
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_y(angle_radians: float) -> np.ndarray:
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_z(angle_radians: float) -> np.ndarray:
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def scale_matrix(scale: float) -> np.ndarray:
    return np.array(
        [
            [scale, 0.0, 0.0, 0.0],
            [0.0, scale, 0.0, 0.0],
            [0.0, 0.0, scale, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def compute_mvp(width: int, height: int, *, yaw: float, pitch: float, roll: float, zoom: float) -> np.ndarray:
    aspect = max(width / max(height, 1), 1e-6)
    projection = perspective_matrix(math.radians(48.0), aspect, 0.1, 100.0)
    view = translation_matrix(0.0, 0.0, -CAMERA_DISTANCE)
    model = rotation_z(math.radians(roll)) @ rotation_x(math.radians(pitch)) @ rotation_y(math.radians(yaw)) @ scale_matrix(zoom)
    return projection @ view @ model


def animated_points(points: np.ndarray, time_value: float) -> np.ndarray:
    if len(points) == 0:
        return points
    pulse = np.float32(1.0 + 0.045 * math.sin(time_value * 0.85))
    drift_angle = math.sin(time_value * 0.33) * 0.32
    drift = rotation_y(drift_angle)[:3, :3]
    return (points * pulse) @ drift.T


def project_ndc(points: np.ndarray, mvp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.empty((points.shape[0], 4), dtype=np.float32)
    homogeneous[:, :3] = points
    homogeneous[:, 3] = 1.0
    clip = homogeneous @ mvp.T
    w = clip[:, 3]
    visible = w > 0.0
    if not np.any(visible):
        return np.empty((0, 3), dtype=np.float32), visible
    ndc = clip[visible, :3] / w[visible, None]
    return ndc.astype(np.float32, copy=False), visible
