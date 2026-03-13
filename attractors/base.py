from __future__ import annotations

from abc import ABC
import random
from typing import Iterable

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator


class Attractor(ABC):
    name = "Base"
    color = (255, 255, 255)
    scale_hint = 1.0
    kernel_derivative = None
    kernel_step = None
    kernel_fill = None
    kernel_sample = None

    def __init__(self) -> None:
        self.state = np.array(self.initial_state(), dtype=np.float64)

    def initial_state(self) -> Iterable[float]:
        return (0.1, 0.0, 0.0)

    def kernel_params(self) -> tuple[float, ...]:
        return ()

    def parameter_dict(self) -> dict[str, float]:
        return {
            key: value
            for key, value in vars(self).items()
            if key != "state" and not key.startswith("_") and isinstance(value, (int, float, np.integer, np.floating))
        }

    def clone(self) -> "Attractor":
        clone = self.__class__(**self.parameter_dict())
        clone.state = self.state.copy()
        return clone

    @classmethod
    def default_parameter_dict(cls) -> dict[str, float]:
        return cls().parameter_dict()

    def reset(self) -> None:
        self.state = np.array(self.initial_state(), dtype=np.float64)

    def set_state(self, state: Iterable[float]) -> None:
        self.state = np.array(tuple(state), dtype=np.float64)

    def set_parameters(self, parameters: dict[str, float]) -> None:
        for key, value in parameters.items():
            setattr(self, key, float(value))

    def randomize_parameters(self, variation: float = 0.2, rng: random.Random | None = None) -> dict[str, float]:
        generator = rng or random.Random()
        randomized: dict[str, float] = {}
        for key, default_value in self.default_parameter_dict().items():
            magnitude = abs(float(default_value))
            if magnitude < 1e-12:
                randomized[key] = float(default_value)
                continue
            spread = magnitude * max(0.0, float(variation))
            randomized[key] = float(generator.uniform(default_value - spread, default_value + spread))
        self.set_parameters(randomized)
        return randomized

    def derivative(self, state: np.ndarray) -> np.ndarray:
        if self.kernel_derivative is None:
            raise NotImplementedError(f"{self.__class__.__name__} is missing kernel_derivative")
        x, y, z = (float(value) for value in state)
        return np.array(self.kernel_derivative(x, y, z, *self.kernel_params()), dtype=np.float64)

    def step(self, dt: float) -> np.ndarray:
        if self.kernel_step is None:
            raise NotImplementedError(f"{self.__class__.__name__} is missing kernel_step")
        x, y, z = self.kernel_step(
            float(self.state[0]),
            float(self.state[1]),
            float(self.state[2]),
            float(dt),
            *self.kernel_params(),
        )
        self.state[0] = x
        self.state[1] = y
        self.state[2] = z
        return self.state.copy()

    def fill_samples(self, dt: float, steps: int, out: np.ndarray | None = None) -> np.ndarray:
        if self.kernel_fill is None:
            raise NotImplementedError(f"{self.__class__.__name__} is missing kernel_fill")
        buffer = out if out is not None else np.empty((steps, 3), dtype=np.float32)
        x, y, z = self.kernel_fill(
            buffer,
            float(self.state[0]),
            float(self.state[1]),
            float(self.state[2]),
            float(dt),
            int(steps),
            *self.kernel_params(),
        )
        self.state[0] = x
        self.state[1] = y
        self.state[2] = z
        return buffer

    def sample_points(
        self,
        count: int,
        *,
        dt: float,
        burn_in: int = 0,
        sample_stride: int = 1,
        out: np.ndarray | None = None,
        initial_state: Iterable[float] | None = None,
        update_state: bool = False,
    ) -> np.ndarray:
        if self.kernel_sample is None:
            raise NotImplementedError(f"{self.__class__.__name__} is missing kernel_sample")
        if count < 1:
            return np.empty((0, 3), dtype=np.float32)
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")

        start_state = tuple(initial_state) if initial_state is not None else tuple(float(value) for value in self.state)
        buffer = out if out is not None else np.empty((count, 3), dtype=np.float32)
        end_x, end_y, end_z = self.kernel_sample(
            buffer,
            float(start_state[0]),
            float(start_state[1]),
            float(start_state[2]),
            float(dt),
            int(burn_in),
            int(sample_stride),
            *self.kernel_params(),
        )
        if update_state:
            self.state[0] = end_x
            self.state[1] = end_y
            self.state[2] = end_z
        return buffer
