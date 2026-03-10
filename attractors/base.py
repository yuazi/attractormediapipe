from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy as np


class Attractor(ABC):
    name = "Base"
    color = (255, 255, 255)
    scale_hint = 1.0

    def __init__(self) -> None:
        self.state = np.array(self.initial_state(), dtype=np.float64)

    def initial_state(self) -> Iterable[float]:
        return (0.1, 0.0, 0.0)

    @abstractmethod
    def derivative(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        self.state = np.array(self.initial_state(), dtype=np.float64)

    def step(self, dt: float) -> np.ndarray:
        k1 = self.derivative(self.state)
        k2 = self.derivative(self.state + dt * k1 * 0.5)
        k3 = self.derivative(self.state + dt * k2 * 0.5)
        k4 = self.derivative(self.state + dt * k3)
        self.state = self.state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.state.copy()
