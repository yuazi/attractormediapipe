from __future__ import annotations

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def rossler_derivative(x: float, y: float, z: float, a: float, b: float, c: float) -> tuple[float, float, float]:
    return (
        -(y + z),
        x + a * y,
        b + z * (x - c),
    )


@njit(cache=True, fastmath=True)
def rossler_rk4_step(x: float, y: float, z: float, dt: float, a: float, b: float, c: float) -> tuple[float, float, float]:
    k1x, k1y, k1z = rossler_derivative(x, y, z, a, b, c)
    k2x, k2y, k2z = rossler_derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, a, b, c)
    k3x, k3y, k3z = rossler_derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, a, b, c)
    k4x, k4y, k4z = rossler_derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z, a, b, c)
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def rossler_fill(out, x: float, y: float, z: float, dt: float, steps: int, a: float, b: float, c: float):
    for idx in range(steps):
        x, y, z = rossler_rk4_step(x, y, z, dt, a, b, c)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def rossler_sample(out, x: float, y: float, z: float, dt: float, burn_in: int, sample_stride: int, a: float, b: float, c: float):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = rossler_rk4_step(x, y, z, dt, a, b, c)
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


class RosslerAttractor(Attractor):
    name = "Rossler"
    color = (255, 215, 90)
    scale_hint = 1.2
    kernel_derivative = staticmethod(rossler_derivative)
    kernel_step = staticmethod(rossler_rk4_step)
    kernel_fill = staticmethod(rossler_fill)
    kernel_sample = staticmethod(rossler_sample)

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7) -> None:
        self.a = a
        self.b = b
        self.c = c
        super().__init__()

    def kernel_params(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)
