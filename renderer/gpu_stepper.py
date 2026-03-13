from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ATTRACTOR_IDS: dict[str, int] = {
    "Lorenz": 0,
    "Aizawa": 1,
    "Sprott B": 2,
    "Thomas": 3,
    "Dadras": 4,
    "Chen": 5,
    "Langford": 6,
    "Rossler": 7,
    "Halvorsen": 8,
}

ATTRACTOR_PARAM_ORDER: dict[str, tuple[str, ...]] = {
    "Lorenz": ("sigma", "rho", "beta"),
    "Aizawa": ("a", "b", "c", "d", "e", "f"),
    "Sprott B": ("a", "b"),
    "Thomas": ("b",),
    "Dadras": ("p", "q", "r", "c", "e"),
    "Chen": ("a", "b", "c"),
    "Langford": ("alpha", "beta", "lam", "omega", "rho", "epsilon"),
    "Rossler": ("a", "b", "c"),
    "Halvorsen": ("a",),
}

MAX_GPU_PARAMS = 6


TRANSFORM_VERTEX_SHADER = """
#version 330

uniform int u_attractor_id;
uniform vec3 u_state;
uniform float u_dt;
uniform float u_params[6];

out vec3 out_position;

vec3 derivative(vec3 state) {
    float x = state.x;
    float y = state.y;
    float z = state.z;

    switch (u_attractor_id) {
        case 0:
            return vec3(
                u_params[0] * (y - x),
                x * (u_params[1] - z) - y,
                x * y - u_params[2] * z
            );
        case 1:
            return vec3(
                (z - u_params[1]) * x - u_params[3] * y,
                u_params[3] * x + (z - u_params[1]) * y,
                u_params[2] + u_params[0] * z - (z * z * z) / 3.0 - (x * x + y * y) * (1.0 + u_params[4] * z) + u_params[5] * z * x * x * x
            );
        case 2:
            return vec3(
                u_params[0] * y * z,
                x - u_params[1] * y,
                1.0 - x * y
            );
        case 3:
            return vec3(
                sin(y) - u_params[0] * x,
                sin(z) - u_params[0] * y,
                sin(x) - u_params[0] * z
            );
        case 4:
            return vec3(
                y - u_params[0] * x + u_params[1] * y * z,
                u_params[2] * y - x * z + z,
                u_params[3] * x * y - u_params[4] * z
            );
        case 5:
            return vec3(
                u_params[0] * (y - x),
                (u_params[2] - u_params[0]) * x - x * z + u_params[2] * y,
                x * y - u_params[1] * z
            );
        case 6:
            return vec3(
                (z - u_params[1]) * x - u_params[3] * y,
                u_params[3] * x + (z - u_params[1]) * y,
                u_params[2] + u_params[0] * z - (z * z * z) / 3.0 - (x * x + y * y) * (1.0 + u_params[4] * z) + u_params[5] * z * x * x * x
            );
        case 7:
            return vec3(
                -(y + z),
                x + u_params[0] * y,
                u_params[1] + z * (x - u_params[2])
            );
        case 8:
            return vec3(
                -u_params[0] * x - 4.0 * y - 4.0 * z - y * y,
                -u_params[0] * y - 4.0 * z - 4.0 * x - z * z,
                -u_params[0] * z - 4.0 * x - 4.0 * y - x * x
            );
    }

    return vec3(0.0);
}

vec3 rk4_step(vec3 state) {
    vec3 k1 = derivative(state);
    vec3 k2 = derivative(state + 0.5 * u_dt * k1);
    vec3 k3 = derivative(state + 0.5 * u_dt * k2);
    vec3 k4 = derivative(state + u_dt * k3);
    return state + (u_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

void main() {
    vec3 state = u_state;
    for (int step_index = 0; step_index <= gl_VertexID; step_index++) {
        state = rk4_step(state);
    }
    out_position = state;
}
"""


def padded_params(attractor_name: str, parameters: dict[str, float]) -> tuple[float, ...]:
    order = ATTRACTOR_PARAM_ORDER[attractor_name]
    values = [float(parameters[key]) for key in order]
    values.extend([0.0] * (MAX_GPU_PARAMS - len(values)))
    return tuple(values[:MAX_GPU_PARAMS])


def rk4_step_cpu(attractor_name: str, state: np.ndarray, dt: float, parameters: dict[str, float]) -> np.ndarray:
    params = padded_params(attractor_name, parameters)
    x, y, z = (float(value) for value in state)

    def derivative(px: float, py: float, pz: float) -> tuple[float, float, float]:
        if attractor_name == "Lorenz":
            return params[0] * (py - px), px * (params[1] - pz) - py, px * py - params[2] * pz
        if attractor_name == "Aizawa":
            return (
                (pz - params[1]) * px - params[3] * py,
                params[3] * px + (pz - params[1]) * py,
                params[2] + params[0] * pz - (pz * pz * pz) / 3.0 - (px * px + py * py) * (1.0 + params[4] * pz) + params[5] * pz * px * px * px,
            )
        if attractor_name == "Sprott B":
            return params[0] * py * pz, px - params[1] * py, 1.0 - px * py
        if attractor_name == "Thomas":
            return np.sin(py), np.sin(pz), np.sin(px)
        if attractor_name == "Dadras":
            return py - params[0] * px + params[1] * py * pz, params[2] * py - px * pz + pz, params[3] * px * py - params[4] * pz
        if attractor_name == "Chen":
            return params[0] * (py - px), (params[2] - params[0]) * px - px * pz + params[2] * py, px * py - params[1] * pz
        if attractor_name == "Langford":
            return (
                (pz - params[1]) * px - params[3] * py,
                params[3] * px + (pz - params[1]) * py,
                params[2] + params[0] * pz - (pz * pz * pz) / 3.0 - (px * px + py * py) * (1.0 + params[4] * pz) + params[5] * pz * px * px * px,
            )
        if attractor_name == "Rossler":
            return -(py + pz), px + params[0] * py, params[1] + pz * (px - params[2])
        if attractor_name == "Halvorsen":
            return (
                -params[0] * px - 4.0 * py - 4.0 * pz - py * py,
                -params[0] * py - 4.0 * pz - 4.0 * px - pz * pz,
                -params[0] * pz - 4.0 * px - 4.0 * py - px * px,
            )
        raise KeyError(attractor_name)

    if attractor_name == "Thomas":
        def derivative(px: float, py: float, pz: float) -> tuple[float, float, float]:
            return np.sin(py) - params[0] * px, np.sin(pz) - params[0] * py, np.sin(px) - params[0] * pz

    k1x, k1y, k1z = derivative(x, y, z)
    k2x, k2y, k2z = derivative(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)
    k3x, k3y, k3z = derivative(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)
    k4x, k4y, k4z = derivative(x + dt * k3x, y + dt * k3y, z + dt * k3z)
    factor = dt / 6.0
    return np.array(
        (
            x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
            y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
        ),
        dtype=np.float32,
    )


def generate_cpu_samples(attractor_name: str, state: tuple[float, float, float], dt: float, parameters: dict[str, float], steps: int) -> np.ndarray:
    if steps <= 0:
        return np.empty((0, 3), dtype=np.float32)
    samples = np.empty((steps, 3), dtype=np.float32)
    current = np.asarray(state, dtype=np.float32)
    for index in range(steps):
        current = rk4_step_cpu(attractor_name, current, dt, parameters)
        samples[index] = current
    return samples


@dataclass
class TransformFeedbackTrailStepper:
    ctx: object

    def __post_init__(self) -> None:
        self._program = self.ctx.program(vertex_shader=TRANSFORM_VERTEX_SHADER, varyings=["out_position"])
        self._vao = self.ctx.vertex_array(self._program, [])
        self._buffer = self.ctx.buffer(reserve=12)
        self._capacity_bytes = 12

    @staticmethod
    def create_if_supported(ctx) -> "TransformFeedbackTrailStepper | None":
        version_code = int(getattr(ctx, "version_code", 0))
        if version_code < 330:
            return None
        try:
            return TransformFeedbackTrailStepper(ctx)
        except Exception:
            return None

    def _ensure_capacity(self, steps: int) -> None:
        byte_count = max(12, int(steps) * 12)
        if byte_count <= self._capacity_bytes:
            return
        self._buffer.release()
        self._buffer = self.ctx.buffer(reserve=byte_count)
        self._capacity_bytes = byte_count

    def generate(
        self,
        *,
        attractor_name: str,
        state: tuple[float, float, float],
        dt: float,
        parameters: dict[str, float],
        steps: int,
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        if steps <= 0:
            return np.empty((0, 3), dtype=np.float32), tuple(float(value) for value in state)

        self._ensure_capacity(steps)
        self._program["u_attractor_id"].value = ATTRACTOR_IDS[attractor_name]
        self._program["u_state"].value = tuple(float(value) for value in state)
        self._program["u_dt"].value = float(dt)
        self._program["u_params"].value = padded_params(attractor_name, parameters)
        self._vao.transform(self._buffer, vertices=steps, mode=self.ctx.POINTS)
        samples = np.frombuffer(self._buffer.read(size=steps * 12), dtype=np.float32).reshape(steps, 3).copy()
        final_state = tuple(float(value) for value in samples[-1])
        return samples, final_state

    def release(self) -> None:
        self._buffer.release()
        self._vao.release()
        self._program.release()
