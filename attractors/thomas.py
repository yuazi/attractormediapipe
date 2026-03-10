from __future__ import annotations

import math

import numpy as np

from .base import Attractor


class ThomasAttractor(Attractor):
    name = "Thomas"
    color = (240, 244, 255)
    scale_hint = 1.5

    def __init__(self, b: float = 0.208) -> None:
        self.b = b
        super().__init__()

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                math.sin(y) - self.b * x,
                math.sin(z) - self.b * y,
                math.sin(x) - self.b * z,
            ],
            dtype=np.float64,
        )
