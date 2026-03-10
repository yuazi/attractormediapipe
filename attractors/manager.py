from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from config import CAMERA_DISTANCE, CAMERA_FOV, TRAIL_BUFFER_CAPACITY

from .aizawa import AizawaAttractor
from .chen import ChenAttractor
from .dadras import DadrasAttractor
from .halvorsen import HalvorsenAttractor
from .lorenz import LorenzAttractor
from .rossler import RosslerAttractor
from .sprott_b import SprottBAttractor
from .thomas import ThomasAttractor


ATTRACTOR_TYPES = (
    LorenzAttractor,
    RosslerAttractor,
    HalvorsenAttractor,
    ThomasAttractor,
    DadrasAttractor,
    AizawaAttractor,
    SprottBAttractor,
    ChenAttractor,
)

PLACARD_MEDIUM = "Generative computation, real-time\nrendering on custom software"


@dataclass(frozen=True)
class PlacardData:
    title: str
    year: str
    medium: str
    params: Tuple[Tuple[str, str], ...]


PLACARD_OVERRIDES = {
    "Lorenz": PlacardData(
        title="Lorenz Attractor",
        year="E. N. Lorenz, 1963",
        medium=PLACARD_MEDIUM,
        params=(("σ (sigma)", "10.000"), ("ρ (rho)", "28.000"), ("β (beta)", "2.667")),
    ),
    "Rossler": PlacardData(
        title="Rössler Attractor",
        year="O. E. Rössler, 1976",
        medium=PLACARD_MEDIUM,
        params=(("a", "0.200"), ("b", "0.200"), ("c", "5.700")),
    ),
    "Halvorsen": PlacardData(
        title="Halvorsen Attractor",
        year="Per Frode Halvorsen",
        medium=PLACARD_MEDIUM,
        params=(("a", "1.890"), ("symmetry", "cyclic"), ("dim", "3")),
    ),
    "Thomas": PlacardData(
        title="Thomas’ Cyclically Symmetric Attractor",
        year="René Thomas, 1999",
        medium=PLACARD_MEDIUM,
        params=(("b", "0.208"), ("symmetry", "cyclic 3-fold"), ("dim", "3")),
    ),
    "Sprott B": PlacardData(
        title="Sprott B Attractor",
        year="J. C. Sprott, 1994",
        medium=PLACARD_MEDIUM,
        params=(("type", "conservative"), ("class", "B"), ("dim", "3")),
    ),
    "Chen": PlacardData(
        title="Chen Attractor",
        year="G. Chen and T. Ueta, 1999",
        medium=PLACARD_MEDIUM,
        params=(("a", "35.000"), ("b", "3.000"), ("c", "28.000")),
    ),
}


def normalize_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    centered = points - points.mean(axis=0, keepdims=True)
    max_extent = np.max(np.abs(centered))
    if max_extent < 1e-8:
        return centered
    return centered / max_extent


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    yaw_r = math.radians(yaw)
    pitch_r = math.radians(pitch)
    roll_r = math.radians(roll)

    rot_y = np.array(
        [
            [math.cos(yaw_r), 0.0, math.sin(yaw_r)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw_r), 0.0, math.cos(yaw_r)],
        ],
        dtype=np.float64,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch_r), -math.sin(pitch_r)],
            [0.0, math.sin(pitch_r), math.cos(pitch_r)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [math.cos(roll_r), -math.sin(roll_r), 0.0],
            [math.sin(roll_r), math.cos(roll_r), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_x @ rot_y


def perspective_project(
    points: np.ndarray,
    rotation: np.ndarray,
    zoom: float,
    screen_size: Tuple[int, int],
    fov: float = CAMERA_FOV,
    camera_distance: float = CAMERA_DISTANCE,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    width, height = screen_size
    rotated = points @ rotation.T
    z = rotated[:, 2] + camera_distance
    z = np.clip(z, 0.2, None)
    sx = rotated[:, 0] * zoom * fov / z + width * 0.5
    sy = rotated[:, 1] * zoom * fov / z + height * 0.5
    depths = np.clip(1.0 - (z - 0.2) / (camera_distance + 2.0), 0.15, 1.0)
    return np.stack((sx, sy), axis=1), depths


class AttractorManager:
    def __init__(self, capacity: int = TRAIL_BUFFER_CAPACITY) -> None:
        self.capacity = capacity
        self._trail = np.zeros((capacity, 3), dtype=np.float64)
        self._count = 0
        self._head = 0
        self.index = 0
        self.current = ATTRACTOR_TYPES[self.index]()

    @property
    def count(self) -> int:
        return self._count

    @property
    def name(self) -> str:
        return self.current.name

    @property
    def color(self) -> Tuple[int, int, int]:
        return self.current.color

    @property
    def scale_hint(self) -> float:
        return self.current.scale_hint

    @property
    def total(self) -> int:
        return len(ATTRACTOR_TYPES)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(attractor_type.name for attractor_type in ATTRACTOR_TYPES)

    @property
    def state_vector(self) -> Tuple[float, float, float]:
        return tuple(float(value) for value in self.current.state)

    @property
    def parameter_rows(self) -> Tuple[Tuple[str, float], ...]:
        rows: list[Tuple[str, float]] = []
        for key, value in vars(self.current).items():
            if key == "state" or key.startswith("_"):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                rows.append((key.upper(), float(value)))
        return tuple(rows)

    @property
    def placard(self) -> PlacardData:
        if self.name in PLACARD_OVERRIDES:
            return PLACARD_OVERRIDES[self.name]

        fallback_rows = tuple((label.lower(), f"{value:0.3f}") for label, value in self.parameter_rows[:3])
        return PlacardData(
            title=f"{self.name} Attractor",
            year="Generative system study",
            medium=PLACARD_MEDIUM,
            params=fallback_rows,
        )

    def reset(self) -> None:
        self.current.reset()
        self.clear_trail()

    def clear_trail(self) -> None:
        self._count = 0
        self._head = 0
        self._trail.fill(0.0)

    def switch_to(self, index: int) -> None:
        new_index = max(0, min(index, self.total - 1))
        if new_index == self.index:
            self.reset()
            return
        self.index = new_index
        self.current = ATTRACTOR_TYPES[self.index]()
        self.clear_trail()

    def switch_relative(self, delta: int) -> None:
        if delta == 0:
            return
        self.index = (self.index + delta) % self.total
        self.current = ATTRACTOR_TYPES[self.index]()
        self.clear_trail()

    def step_many(self, dt: float, steps: int) -> np.ndarray:
        samples = np.zeros((steps, 3), dtype=np.float64)
        for idx in range(steps):
            point = self.current.step(dt)
            self._append_point(point)
            samples[idx] = point
        return samples

    def _append_point(self, point: np.ndarray) -> None:
        self._trail[self._head] = point
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def get_recent_trail(self, limit: int | None = None) -> np.ndarray:
        if self._count == 0:
            return np.empty((0, 3), dtype=np.float64)
        limit = self._count if limit is None else min(limit, self._count)
        start = (self._head - limit) % self.capacity
        if start < self._head and limit == self._head - start:
            return self._trail[start:self._head].copy()
        indices = (np.arange(limit) + start) % self.capacity
        return self._trail[indices].copy()

    def get_projected_trail(
        self,
        limit: int,
        yaw: float,
        pitch: float,
        roll: float,
        zoom: float,
        screen_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        trail = self.get_recent_trail(limit)
        if len(trail) == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
        normalized = normalize_points(trail)
        rotation = rotation_matrix(yaw, pitch, roll)
        return perspective_project(normalized, rotation, zoom * self.scale_hint, screen_size)
