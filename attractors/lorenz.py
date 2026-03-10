from __future__ import annotations

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def lorenz_derivative(x: float, y: float, z: float, sigma: float, rho: float, beta: float) -> tuple[float, float, float]:
    return (
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    )


@njit(cache=True, fastmath=True)
def lorenz_rk4_step(x: float, y: float, z: float, dt: float, sigma: float, rho: float, beta: float) -> tuple[float, float, float]:
    k1x, k1y, k1z = lorenz_derivative(x, y, z, sigma, rho, beta)
    k2x, k2y, k2z = lorenz_derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z, sigma, rho, beta)
    k3x, k3y, k3z = lorenz_derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z, sigma, rho, beta)
    k4x, k4y, k4z = lorenz_derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z, sigma, rho, beta)
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def lorenz_fill(out, x: float, y: float, z: float, dt: float, steps: int, sigma: float, rho: float, beta: float):
    for idx in range(steps):
        x, y, z = lorenz_rk4_step(x, y, z, dt, sigma, rho, beta)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def lorenz_sample(out, x: float, y: float, z: float, dt: float, burn_in: int, sample_stride: int, sigma: float, rho: float, beta: float):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = lorenz_rk4_step(x, y, z, dt, sigma, rho, beta)
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


class LorenzAttractor(Attractor):
    name = "Lorenz"
    color = (255, 88, 88)
    scale_hint = 1.0
    kernel_derivative = staticmethod(lorenz_derivative)
    kernel_step = staticmethod(lorenz_rk4_step)
    kernel_fill = staticmethod(lorenz_fill)
    kernel_sample = staticmethod(lorenz_sample)

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super().__init__()

    def kernel_params(self) -> tuple[float, float, float]:
        return (self.sigma, self.rho, self.beta)
