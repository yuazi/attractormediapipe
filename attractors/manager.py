from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from config import DEFAULT_SCALE, TRAIL_BUFFER_CAPACITY

from .aizawa import AizawaAttractor
from .chen import ChenAttractor
from .dadras import DadrasAttractor
from .halvorsen import HalvorsenAttractor
from .langford import LangfordAttractor
from .lorenz import LorenzAttractor
from .rossler import RosslerAttractor
from .sprott_b import SprottBAttractor
from .thomas import ThomasAttractor


ACTIVE_ATTRACTOR_TYPES = (
    LorenzAttractor,
    AizawaAttractor,
    SprottBAttractor,
    ThomasAttractor,
    DadrasAttractor,
    ChenAttractor,
    LangfordAttractor,
)

INACTIVE_ATTRACTOR_TYPES = (
    RosslerAttractor,
    HalvorsenAttractor,
)

PLACARD_MEDIUM = "Generative computation, real-time\nrendering on custom software"


@dataclass(frozen=True)
class PlacardData:
    title: str
    year: str
    medium: str
    params: tuple[tuple[str, str], ...]


PLACARD_OVERRIDES = {
    "Lorenz": PlacardData(
        title="Lorenz Attractor",
        year="E. N. Lorenz, 1963",
        medium=PLACARD_MEDIUM,
        params=(("sigma", "10.000"), ("rho", "28.000"), ("beta", "2.667")),
    ),
    "Aizawa": PlacardData(
        title="Aizawa Attractor",
        year="K. Aizawa, 1982",
        medium=PLACARD_MEDIUM,
        params=(("a", "0.950"), ("b", "0.700"), ("c", "0.600")),
    ),
    "Sprott B": PlacardData(
        title="Sprott B Attractor",
        year="J. C. Sprott, 1994",
        medium=PLACARD_MEDIUM,
        params=(("a", "1.000"), ("b", "1.000"), ("class", "B")),
    ),
    "Thomas": PlacardData(
        title="Thomas' Attractor",
        year="Rene Thomas, 1999",
        medium=PLACARD_MEDIUM,
        params=(("b", "0.208"), ("symmetry", "cyclic"), ("dim", "3")),
    ),
    "Dadras": PlacardData(
        title="Dadras Attractor",
        year="S. Dadras et al., 2006",
        medium=PLACARD_MEDIUM,
        params=(("p", "3.000"), ("q", "2.700"), ("r", "1.700")),
    ),
    "Chen": PlacardData(
        title="Chen Attractor",
        year="G. Chen and T. Ueta, 1999",
        medium=PLACARD_MEDIUM,
        params=(("a", "35.000"), ("b", "3.000"), ("c", "28.000")),
    ),
    "Langford": PlacardData(
        title="Langford Attractor",
        year="W. F. Langford",
        medium=PLACARD_MEDIUM,
        params=(("alpha", "0.950"), ("beta", "0.700"), ("omega", "3.500")),
    ),
}


def normalize_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    centered = points.astype(np.float32, copy=False) - points.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    max_extent = float(np.max(np.abs(centered)))
    if max_extent < 1e-6:
        return centered
    return centered / np.float32(max_extent)


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
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch_r), -math.sin(pitch_r)],
            [0.0, math.sin(pitch_r), math.cos(pitch_r)],
        ],
        dtype=np.float32,
    )
    rot_z = np.array(
        [
            [math.cos(roll_r), -math.sin(roll_r), 0.0],
            [math.sin(roll_r), math.cos(roll_r), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rot_z @ rot_x @ rot_y


def perspective_project(
    points: np.ndarray,
    rotation: np.ndarray,
    zoom: float,
    screen_size: tuple[int, int],
    *,
    fov: float = 520.0,
    camera_distance: float = 4.6,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    width, height = screen_size
    rotated = points @ rotation.T
    z = rotated[:, 2] + camera_distance
    z = np.clip(z, 0.2, None)
    sx = rotated[:, 0] * zoom * fov / z + width * 0.5
    sy = rotated[:, 1] * zoom * fov / z + height * 0.5
    depths = np.clip(1.0 - (z - 0.2) / (camera_distance + 2.0), 0.15, 1.0)
    return np.stack((sx, sy), axis=1).astype(np.float32, copy=False), depths.astype(np.float32, copy=False)


class AttractorManager:
    def __init__(self, capacity: int = TRAIL_BUFFER_CAPACITY) -> None:
        self.capacity = capacity
        self.attractors = [attractor_type() for attractor_type in ACTIVE_ATTRACTOR_TYPES]
        self._trails = [np.zeros((capacity, 3), dtype=np.float32) for _ in self.attractors]
        self._counts = np.zeros(len(self.attractors), dtype=np.int32)
        self._heads = np.zeros(len(self.attractors), dtype=np.int32)
        self.index = 0
        self.zoom = DEFAULT_SCALE

    @property
    def current(self):
        return self.attractors[self.index]

    @property
    def count(self) -> int:
        return int(self._counts[self.index])

    @property
    def name(self) -> str:
        return self.current.name

    @property
    def color(self) -> tuple[int, int, int]:
        return self.current.color

    @property
    def scale_hint(self) -> float:
        return self.current.scale_hint

    @property
    def total(self) -> int:
        return len(self.attractors)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(attractor.name for attractor in self.attractors)

    @property
    def state_vector(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.current.state)

    @property
    def parameter_rows(self) -> tuple[tuple[str, float], ...]:
        return tuple((key.upper(), float(value)) for key, value in self.current.parameter_dict().items())

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

    def active_index_for_name(self, name: str) -> int:
        normalized = name.strip().lower()
        for index, attractor in enumerate(self.attractors):
            if attractor.name.lower() == normalized:
                return index
        raise KeyError(name)

    def get_attractor(self, index: int | None = None):
        target_index = self.index if index is None else max(0, min(index, self.total - 1))
        return self.attractors[target_index]

    def clone_current(self):
        return self.current.clone()

    def reset(self) -> None:
        self.current.reset()
        self.clear_trail(self.index)

    def reset_all(self) -> None:
        for index, attractor in enumerate(self.attractors):
            attractor.reset()
            self.clear_trail(index)

    def clear_trail(self, index: int | None = None) -> None:
        target_index = self.index if index is None else max(0, min(index, self.total - 1))
        self._counts[target_index] = 0
        self._heads[target_index] = 0
        self._trails[target_index].fill(0.0)

    def switch_to(self, index: int) -> None:
        new_index = max(0, min(index, self.total - 1))
        if new_index == self.index:
            self.reset()
            return
        self.index = new_index
        self.current.reset()
        self.clear_trail(new_index)

    def switch_relative(self, delta: int) -> None:
        if delta == 0:
            return
        self.index = (self.index + delta) % self.total
        self.current.reset()
        self.clear_trail(self.index)

    def step_many(self, dt: float, steps: int) -> np.ndarray:
        samples = self.current.fill_samples(dt, steps)
        self._append_points(self.index, samples)
        return samples

    def _append_points(self, index: int, points: np.ndarray) -> None:
        if len(points) == 0:
            return
        head = int(self._heads[index])
        count = len(points)
        first_chunk = min(self.capacity - head, count)
        trail = self._trails[index]
        trail[head:head + first_chunk] = points[:first_chunk]
        remaining = count - first_chunk
        if remaining > 0:
            trail[:remaining] = points[first_chunk:]
        self._heads[index] = (head + count) % self.capacity
        self._counts[index] = min(self.capacity, int(self._counts[index]) + count)

    def get_recent_trail(self, limit: int | None = None, *, index: int | None = None) -> np.ndarray:
        target_index = self.index if index is None else max(0, min(index, self.total - 1))
        count = int(self._counts[target_index])
        if count == 0:
            return np.empty((0, 3), dtype=np.float32)

        take = count if limit is None else min(limit, count)
        head = int(self._heads[target_index])
        start = (head - take) % self.capacity
        trail = self._trails[target_index]
        if start < head and take == head - start:
            return trail[start:head].copy()
        indices = (np.arange(take, dtype=np.int32) + start) % self.capacity
        return trail[indices].copy()

    def get_render_data(self, limit: int) -> tuple[np.ndarray, np.ndarray]:
        trail = self.get_recent_trail(limit)
        normalized = normalize_points(trail)
        if len(normalized) == 0:
            return normalized, np.empty((0,), dtype=np.float32)
        normalized *= np.float32(self.scale_hint)
        ages = np.linspace(0.02, 1.0, num=len(normalized), dtype=np.float32)
        return normalized.astype(np.float32, copy=False), ages

    def get_projected_trail(
        self,
        limit: int,
        yaw: float,
        pitch: float,
        roll: float,
        zoom: float,
        screen_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        trail = self.get_recent_trail(limit)
        if len(trail) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        normalized = normalize_points(trail) * np.float32(self.scale_hint)
        rotation = rotation_matrix(yaw, pitch, roll)
        return perspective_project(normalized, rotation, zoom, screen_size)


def create_active_attractor(name: str):
    normalized = name.strip().lower()
    for attractor_type in ACTIVE_ATTRACTOR_TYPES:
        if attractor_type.name.lower() == normalized:
            return attractor_type()
    raise KeyError(name)


def active_attractor_names() -> Sequence[str]:
    return tuple(attractor_type.name for attractor_type in ACTIVE_ATTRACTOR_TYPES)


def inactive_attractor_names() -> Iterable[str]:
    return tuple(attractor_type.name for attractor_type in INACTIVE_ATTRACTOR_TYPES)
