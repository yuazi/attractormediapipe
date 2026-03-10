from __future__ import annotations

import math

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def thomas_derivative(x: float, y: float, z: float, b: float) -> tuple[float, float, float]:
    return (
        math.sin(y) - b * x,
        math.sin(z) - b * y,
        math.sin(x) - b * z,
    )


@njit(cache=True, fastmath=True)
def thomas_rk4_step(x: float, y: float, z: float, dt: float, b: float) -> tuple[float, float, float]:
    k1x, k1y, k1z = thomas_derivative(x, y, z, b)
    k2x, k2y, k2z = thomas_derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, b)
    k3x, k3y, k3z = thomas_derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, b)
    k4x, k4y, k4z = thomas_derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z, b)
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def thomas_fill(out, x: float, y: float, z: float, dt: float, steps: int, b: float):
    for idx in range(steps):
        x, y, z = thomas_rk4_step(x, y, z, dt, b)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def thomas_sample(out, x: float, y: float, z: float, dt: float, burn_in: int, sample_stride: int, b: float):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = thomas_rk4_step(x, y, z, dt, b)
        if step_index < burn_in:
            continue
        if (step_index - burn_in) % sample_stride != 0:
            continue
        out[written, 0] = x
        out[written, 1] = y
        out[written, 2] = z
        written += 1
        if written >= count:
            break
    return x, y, z


class ThomasAttractor(Attractor):
    name = "Thomas"
    color = (98, 255, 229)
    scale_hint = 1.5
    kernel_derivative = staticmethod(thomas_derivative)
    kernel_step = staticmethod(thomas_rk4_step)
    kernel_fill = staticmethod(thomas_fill)
    kernel_sample = staticmethod(thomas_sample)

    def __init__(self, b: float = 0.208) -> None:
        self.b = b
        super().__init__()

    def kernel_params(self) -> tuple[float]:
        return (self.b,)
