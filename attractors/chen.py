from __future__ import annotations

import numpy as np

from .base import Attractor


class ChenAttractor(Attractor):
    name = "Chen"
    color = (255, 122, 90)
    scale_hint = 1.0

    def __init__(self, a: float = 35.0, b: float = 3.0, c: float = 28.0) -> None:
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def initial_state(self):
        return (-0.1, 0.5, -0.6)

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                self.a * (y - x),
                (self.c - self.a) * x - x * z + self.c * y,
                x * y - self.b * z,
            ],
            dtype=np.float64,
        )
