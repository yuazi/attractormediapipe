from __future__ import annotations

import numpy as np

from .base import Attractor


class SprottBAttractor(Attractor):
    name = "Sprott B"
    color = (255, 72, 104)
    scale_hint = 1.2

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self.a = a
        self.b = b
        super().__init__()

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                self.a * y * z,
                x - self.b * y,
                1.0 - x * y,
            ],
            dtype=np.float64,
        )
