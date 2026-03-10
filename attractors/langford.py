from __future__ import annotations

from .base import Attractor, njit


@njit(cache=True, fastmath=True)
def langford_derivative(
    x: float,
    y: float,
    z: float,
    alpha: float,
    beta: float,
    lam: float,
    omega: float,
    rho: float,
    epsilon: float,
) -> tuple[float, float, float]:
    x2 = x * x
    y2 = y * y
    z2 = z * z
    return (
        (z - beta) * x - omega * y,
        omega * x + (z - beta) * y,
        lam + alpha * z - (z * z2) / 3.0 - (x2 + y2) * (1.0 + rho * z) + epsilon * z * (x2 * x),
    )


@njit(cache=True, fastmath=True)
def langford_rk4_step(
    x: float,
    y: float,
    z: float,
    dt: float,
    alpha: float,
    beta: float,
    lam: float,
    omega: float,
    rho: float,
    epsilon: float,
) -> tuple[float, float, float]:
    k1x, k1y, k1z = langford_derivative(x, y, z, alpha, beta, lam, omega, rho, epsilon)
    k2x, k2y, k2z = langford_derivative(
        x + 0.5 * dt * k1x,
        y + 0.5 * dt * k1y,
        z + 0.5 * dt * k1z,
        alpha,
        beta,
        lam,
        omega,
        rho,
        epsilon,
    )
    k3x, k3y, k3z = langford_derivative(
        x + 0.5 * dt * k2x,
        y + 0.5 * dt * k2y,
        z + 0.5 * dt * k2z,
        alpha,
        beta,
        lam,
        omega,
        rho,
        epsilon,
    )
    k4x, k4y, k4z = langford_derivative(
        x + dt * k3x,
        y + dt * k3y,
        z + dt * k3z,
        alpha,
        beta,
        lam,
        omega,
        rho,
        epsilon,
    )
    factor = dt / 6.0
    return (
        x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
        z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
    )


@njit(cache=True, fastmath=True)
def langford_fill(
    out,
    x: float,
    y: float,
    z: float,
    dt: float,
    steps: int,
    alpha: float,
    beta: float,
    lam: float,
    omega: float,
    rho: float,
    epsilon: float,
):
    for idx in range(steps):
        x, y, z = langford_rk4_step(x, y, z, dt, alpha, beta, lam, omega, rho, epsilon)
        out[idx, 0] = x
        out[idx, 1] = y
        out[idx, 2] = z
    return x, y, z


@njit(cache=True, fastmath=True)
def langford_sample(
    out,
    x: float,
    y: float,
    z: float,
    dt: float,
    burn_in: int,
    sample_stride: int,
    alpha: float,
    beta: float,
    lam: float,
    omega: float,
    rho: float,
    epsilon: float,
):
    count = out.shape[0]
    written = 0
    total_steps = burn_in + count * sample_stride
    for step_index in range(total_steps):
        x, y, z = langford_rk4_step(x, y, z, dt, alpha, beta, lam, omega, rho, epsilon)
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


class LangfordAttractor(Attractor):
    name = "Langford"
    color = (255, 86, 171)
    scale_hint = 2.0
    kernel_derivative = staticmethod(langford_derivative)
    kernel_step = staticmethod(langford_rk4_step)
    kernel_fill = staticmethod(langford_fill)
    kernel_sample = staticmethod(langford_sample)

    def __init__(
        self,
        alpha: float = 0.95,
        beta: float = 0.7,
        lam: float = 0.6,
        omega: float = 3.5,
        rho: float = 0.25,
        epsilon: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.omega = omega
        self.rho = rho
        self.epsilon = epsilon
        super().__init__()

    def kernel_params(self) -> tuple[float, float, float, float, float, float]:
        return (self.alpha, self.beta, self.lam, self.omega, self.rho, self.epsilon)
