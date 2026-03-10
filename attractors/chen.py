from __future__ import annotations

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def chen_derivative(x: float, y: float, z: float, a: float, b: float, c: float) -> tuple[float, float, float]:
    return (
        a * (y - x),
        (c - a) * x - x * z + c * y,
        x * y - b * z,
    )


@njit(cache=True, fastmath=True)
def chen_rk4_step(x: float, y: float, z: float, dt: float, a: float, b: float, c: float) -> tuple[float, float, float]:
    k1x, k1y, k1z = chen_derivative(x, y, z, a, b, c)
    k2x, k2y, k2z = chen_derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, a, b, c)
    k3x, k3y, k3z = chen_derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, a, b, c)
    k4x, k4y, k4z = chen_derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z, a, b, c)
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def chen_fill(out, x: float, y: float, z: float, dt: float, steps: int, a: float, b: float, c: float):
    for idx in range(steps):
        x, y, z = chen_rk4_step(x, y, z, dt, a, b, c)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def chen_sample(out, x: float, y: float, z: float, dt: float, burn_in: int, sample_stride: int, a: float, b: float, c: float):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = chen_rk4_step(x, y, z, dt, a, b, c)
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


class ChenAttractor(Attractor):
    name = "Chen"
    color = (77, 115, 255)
    scale_hint = 1.0
    kernel_derivative = staticmethod(chen_derivative)
    kernel_step = staticmethod(chen_rk4_step)
    kernel_fill = staticmethod(chen_fill)
    kernel_sample = staticmethod(chen_sample)

    def __init__(self, a: float = 35.0, b: float = 3.0, c: float = 28.0) -> None:
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def initial_state(self):
        return (-0.1, 0.5, -0.6)

    def kernel_params(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)
