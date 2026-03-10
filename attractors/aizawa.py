from __future__ import annotations

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def aizawa_derivative(
    x: float,
    y: float,
    z: float,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> tuple[float, float, float]:
    x2 = x * x
    y2 = y * y
    z2 = z * z
    return (
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - (z * z2) / 3.0 - (x2 + y2) * (1.0 + e * z) + f * z * (x2 * x),
    )


@njit(cache=True, fastmath=True)
def aizawa_rk4_step(
    x: float,
    y: float,
    z: float,
    dt: float,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> tuple[float, float, float]:
    k1x, k1y, k1z = aizawa_derivative(x, y, z, a, b, c, d, e, f)
    k2x, k2y, k2z = aizawa_derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, a, b, c, d, e, f)
    k3x, k3y, k3z = aizawa_derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, a, b, c, d, e, f)
    k4x, k4y, k4z = aizawa_derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z, a, b, c, d, e, f)
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def aizawa_fill(out, x: float, y: float, z: float, dt: float, steps: int, a: float, b: float, c: float, d: float, e: float, f: float):
    for idx in range(steps):
        x, y, z = aizawa_rk4_step(x, y, z, dt, a, b, c, d, e, f)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def aizawa_sample(out, x: float, y: float, z: float, dt: float, burn_in: int, sample_stride: int, a: float, b: float, c: float, d: float, e: float, f: float):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = aizawa_rk4_step(x, y, z, dt, a, b, c, d, e, f)
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


class AizawaAttractor(Attractor):
    name = "Aizawa"
    color = (255, 177, 74)
    scale_hint = 2.0
    kernel_derivative = staticmethod(aizawa_derivative)
    kernel_step = staticmethod(aizawa_rk4_step)
    kernel_fill = staticmethod(aizawa_fill)
    kernel_sample = staticmethod(aizawa_sample)

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

    def kernel_params(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.d, self.e, self.f)
