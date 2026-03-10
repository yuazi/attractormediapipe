from __future__ import annotations

import numpy as np

from .base import Attractor


class LorenzAttractor(Attractor):
    name = "Lorenz"
    color = (38, 255, 164)
    scale_hint = 1.0

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super().__init__()

    def derivative(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                self.sigma * (y - x),
                x * (self.rho - z) - y,
                x * y - self.beta * z,
            ],
            dtype=np.float64,
        )
