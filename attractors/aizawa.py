from __future__ import annotations

import numpy as np

from .base import Attractor


class AizawaAttractor(Attractor):
    name = "Aizawa"
    color = (235, 238, 255)
    scale_hint = 2.0

    def __init__(
        self,
        a: float = 0.95,
        b: float = 0.7,
        c: float = 0.6,
        d: float = 3.5,
        e: float = 0.25,
        f: float = 0.1,
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        super().__init__()

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                (z - self.b) * x - self.d * y,
                self.d * x + (z - self.b) * y,
                self.c
                + self.a * z
                - (z**3) / 3.0
                - (x * x + y * y) * (1.0 + self.e * z)
                + self.f * z * (x**3),
            ],
            dtype=np.float64,
        )
