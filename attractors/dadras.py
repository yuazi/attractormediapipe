from __future__ import annotations

import numpy as np

from .base import Attractor


class DadrasAttractor(Attractor):
    name = "Dadras"
    color = (173, 102, 255)
    scale_hint = 1.0

    def __init__(self, p: float = 3.0, q: float = 2.7, r: float = 1.7, c: float = 2.0, e: float = 9.0) -> None:
        self.p = p
        self.q = q
        self.r = r
        self.c = c
        self.e = e
        super().__init__()

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                y - self.p * x + self.q * y * z,
                self.r * y - x * z + z,
                self.c * x * y - self.e * z,
            ],
            dtype=np.float64,
        )
