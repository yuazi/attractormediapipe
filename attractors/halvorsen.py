from __future__ import annotations

import numpy as np

from .base import Attractor


class HalvorsenAttractor(Attractor):
    name = "Halvorsen"
    color = (255, 200, 78)
    scale_hint = 0.8

    def __init__(self, a: float = 1.4) -> None:
        self.a = a
        super().__init__()

    def initial_state(self):
        return (1.0, 0.0, 0.0)

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                -self.a * x - 4.0 * y - 4.0 * z - y * y,
                -self.a * y - 4.0 * z - 4.0 * x - z * z,
                -self.a * z - 4.0 * x - 4.0 * y - x * x,
            ],
            dtype=np.float64,
        )
