from __future__ import annotations

import numpy as np

from .base import Attractor


class RosslerAttractor(Attractor):
    name = "Rossler"
    color = (78, 219, 255)
    scale_hint = 1.2

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7) -> None:
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                -(y + z),
                x + self.a * y,
                self.b + z * (x - self.c),
            ],
            dtype=np.float64,
        )
