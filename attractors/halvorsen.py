from __future__ import annotations

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def halvorsen_derivative(x: float, y: float, z: float, a: float) -> tuple[float, float, float]:
    return (
        -a * x - 4.0 * y - 4.0 * z - y * y,
        -a * y - 4.0 * z - 4.0 * x - z * z,
        -a * z - 4.0 * x - 4.0 * y - x * x,
    )


@njit(cache=True, fastmath=True)
def halvorsen_rk4_step(x: float, y: float, z: float, dt: float, a: float) -> tuple[float, float, float]:
    k1x, k1y, k1z = halvorsen_derivative(x, y, z, a)
    k2x, k2y, k2z = halvorsen_derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, a)
    k3x, k3y, k3z = halvorsen_derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, a)
    k4x, k4y, k4z = halvorsen_derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z, a)
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def halvorsen_fill(out, x: float, y: float, z: float, dt: float, steps: int, a: float):
    for idx in range(steps):
        x, y, z = halvorsen_rk4_step(x, y, z, dt, a)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def halvorsen_sample(out, x: float, y: float, z: float, dt: float, burn_in: int, sample_stride: int, a: float):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = halvorsen_rk4_step(x, y, z, dt, a)
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


class HalvorsenAttractor(Attractor):
    name = "Halvorsen"
    color = (255, 203, 98)
    scale_hint = 0.8
    kernel_derivative = staticmethod(halvorsen_derivative)
    kernel_step = staticmethod(halvorsen_rk4_step)
    kernel_fill = staticmethod(halvorsen_fill)
    kernel_sample = staticmethod(halvorsen_sample)

    def __init__(self, a: float = 1.4) -> None:
        self.a = a
        super().__init__()

    def initial_state(self):
        return (1.0, 0.0, 0.0)

    def kernel_params(self) -> tuple[float]:
        return (self.a,)
