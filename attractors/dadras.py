from __future__ import annotations

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def dadras_derivative(x: float, y: float, z: float, p: float, q: float, r: float, c: float, e: float) -> tuple[float, float, float]:
    return (
        y - p * x + q * y * z,
        r * y - x * z + z,
        c * x * y - e * z,
    )


@njit(cache=True, fastmath=True)
def dadras_rk4_step(x: float, y: float, z: float, dt: float, p: float, q: float, r: float, c: float, e: float) -> tuple[float, float, float]:
    k1x, k1y, k1z = dadras_derivative(x, y, z, p, q, r, c, e)
    k2x, k2y, k2z = dadras_derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, p, q, r, c, e)
    k3x, k3y, k3z = dadras_derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, p, q, r, c, e)
    k4x, k4y, k4z = dadras_derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z, p, q, r, c, e)
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def dadras_fill(out, x: float, y: float, z: float, dt: float, steps: int, p: float, q: float, r: float, c: float, e: float):
    for idx in range(steps):
        x, y, z = dadras_rk4_step(x, y, z, dt, p, q, r, c, e)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def dadras_sample(out, x: float, y: float, z: float, dt: float, burn_in: int, sample_stride: int, p: float, q: float, r: float, c: float, e: float):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = dadras_rk4_step(x, y, z, dt, p, q, r, c, e)
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


class DadrasAttractor(Attractor):
    name = "Dadras"
    color = (89, 164, 255)
    scale_hint = 1.0
    kernel_derivative = staticmethod(dadras_derivative)
    kernel_step = staticmethod(dadras_rk4_step)
    kernel_fill = staticmethod(dadras_fill)
    kernel_sample = staticmethod(dadras_sample)

    def __init__(self, p: float = 3.0, q: float = 2.7, r: float = 1.7, c: float = 2.0, e: float = 9.0) -> None:
        self.p = p
        self.q = q
        self.r = r
        self.c = c
        self.e = e
        super().__init__()

    def kernel_params(self) -> tuple[float, float, float, float, float]:
        return (self.p, self.q, self.r, self.c, self.e)
