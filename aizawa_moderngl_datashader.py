#!/usr/bin/env python3
from __future__ import annotations

"""
Real-time Aizawa strange attractor renderer.

Dependencies for this standalone script:
    python3 -m pip install numpy numba pygame moderngl datashader pandas Pillow

Examples:
    python3 aizawa_moderngl_datashader.py
    python3 aizawa_moderngl_datashader.py --export-only --export assets/aizawa_frame.png
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator


AIZAWA_A = 0.95
AIZAWA_B = 0.7
AIZAWA_C = 0.6
AIZAWA_D = 3.5
AIZAWA_E = 0.25
AIZAWA_F = 0.1

DEFAULT_POINT_COUNT = 1_200_000
DEFAULT_DT = 0.0035
DEFAULT_BURN_IN = 20_000
DEFAULT_SAMPLE_STRIDE = 1

VERTEX_SHADER = """
#version 330

uniform mat4 u_mvp;
uniform float u_point_scale;

in vec3 in_position;
in float in_tone;

out float v_tone;

void main() {
    vec4 clip = u_mvp * vec4(in_position, 1.0);
    gl_Position = clip;

    float depth = max(0.35, clip.w);
    gl_PointSize = clamp((u_point_scale / depth) + 0.9, 1.0, 4.25);
    v_tone = clamp(in_tone, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330

in float v_tone;
out vec4 fragColor;

vec3 palette(float t) {
    vec3 low = vec3(0.08, 0.33, 0.98);
    vec3 mid = vec3(0.95, 0.47, 0.15);
    vec3 high = vec3(1.00, 0.95, 0.84);

    float firstMix = smoothstep(0.0, 0.58, t);
    float secondMix = smoothstep(0.58, 1.0, t);
    vec3 ramp = mix(low, mid, firstMix);
    return mix(ramp, high, secondMix);
}

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) {
        discard;
    }

    float halo = exp(-3.8 * r2);
    float core = pow(max(0.0, 1.0 - r2), 6.0);
    float glow = max(halo * 0.65, core);
    float alpha = glow * (0.022 + (0.052 * v_tone));

    fragColor = vec4(palette(v_tone), alpha);
}
"""


@dataclass(frozen=True)
class PointCloud:
    positions: np.ndarray
    tone: np.ndarray
    speeds: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])


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
    dx = (z - b) * x - d * y
    dy = d * x + (z - b) * y
    x2 = x * x
    y2 = y * y
    z2 = z * z
    dz = c + a * z - (z * z2) / 3.0 - (x2 + y2) * (1.0 + e * z) + f * z * (x2 * x)
    return dx, dy, dz


@njit(cache=True, fastmath=True)
def rk4_step(
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

    x2 = x + 0.5 * dt * k1x
    y2 = y + 0.5 * dt * k1y
    z2 = z + 0.5 * dt * k1z
    k2x, k2y, k2z = aizawa_derivative(x2, y2, z2, a, b, c, d, e, f)

    x3 = x + 0.5 * dt * k2x
    y3 = y + 0.5 * dt * k2y
    z3 = z + 0.5 * dt * k2z
    k3x, k3y, k3z = aizawa_derivative(x3, y3, z3, a, b, c, d, e, f)

    x4 = x + dt * k3x
    y4 = y + dt * k3y
    z4 = z + dt * k3z
    k4x, k4y, k4z = aizawa_derivative(x4, y4, z4, a, b, c, d, e, f)

    factor = dt / 6.0
    nx = x + factor * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    ny = y + factor * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    nz = z + factor * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return nx, ny, nz


@njit(cache=True, fastmath=True)
def generate_aizawa_samples(count: int, burn_in: int, dt: float, sample_stride: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((count, 3), dtype=np.float32)
    speeds = np.empty(count, dtype=np.float32)

    x = 0.1
    y = 0.0
    z = 0.0

    samples_written = 0
    total_steps = burn_in + (count * sample_stride)
    for step_index in range(total_steps):
        x, y, z = rk4_step(
            x,
            y,
            z,
            dt,
            AIZAWA_A,
            AIZAWA_B,
            AIZAWA_C,
            AIZAWA_D,
            AIZAWA_E,
            AIZAWA_F,
        )

        if step_index < burn_in:
            continue
        if (step_index - burn_in) % sample_stride != 0:
            continue

        dx, dy, dz = aizawa_derivative(
            x,
            y,
            z,
            AIZAWA_A,
            AIZAWA_B,
            AIZAWA_C,
            AIZAWA_D,
            AIZAWA_E,
            AIZAWA_F,
        )
        speed = math.sqrt(dx * dx + dy * dy + dz * dz)

        positions[samples_written, 0] = x
        positions[samples_written, 1] = y
        positions[samples_written, 2] = z
        speeds[samples_written] = speed
        samples_written += 1

        if samples_written >= count:
            break

    return positions, speeds


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModernGL Aizawa strange attractor renderer")
    parser.add_argument("--points", type=int, default=DEFAULT_POINT_COUNT, help="Number of samples in the point cloud")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="RK4 integration step size")
    parser.add_argument("--burn-in", type=int, default=DEFAULT_BURN_IN, help="Transient integration steps to discard")
    parser.add_argument("--sample-stride", type=int, default=DEFAULT_SAMPLE_STRIDE, help="Sub-sampling factor after burn-in")
    parser.add_argument("--width", type=int, default=1600, help="Window width")
    parser.add_argument("--height", type=int, default=900, help="Window height")
    parser.add_argument("--export-width", type=int, default=4096, help="Datashader export width")
    parser.add_argument("--export-height", type=int, default=4096, help="Datashader export height")
    parser.add_argument("--point-scale", type=float, default=2.75, help="Base point size for gl_PointSize")
    parser.add_argument(
        "--color-mode",
        choices=("velocity", "height", "mix"),
        default="velocity",
        help="Gradient driver for particle color",
    )
    parser.add_argument("--camera-distance", type=float, default=7.6, help="View-space camera distance")
    parser.add_argument("--export", type=Path, default=Path("aizawa_datashader_frame.png"), help="Datashader output path")
    parser.add_argument("--export-only", action="store_true", help="Skip the ModernGL viewer and only save a static frame")
    return parser.parse_args(list(argv) if argv is not None else None)


def normalize_points(positions: np.ndarray) -> np.ndarray:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) * 0.5
    extent = float(np.max(maxs - mins))
    scale = 4.2 / max(extent, 1e-6)
    return (positions - center).astype(np.float32, copy=False) * np.float32(scale)


def normalize_scalar(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = max(maximum - minimum, 1e-6)
    return ((values - minimum) / span).astype(np.float32, copy=False)


def compute_tone(positions: np.ndarray, speeds: np.ndarray, mode: str) -> np.ndarray:
    velocity = normalize_scalar(speeds)
    height = normalize_scalar(positions[:, 2])
    if mode == "velocity":
        return velocity
    if mode == "height":
        return height
    return np.clip((0.6 * velocity) + (0.4 * height), 0.0, 1.0).astype(np.float32, copy=False)


def build_point_cloud(args: argparse.Namespace) -> PointCloud:
    if args.points < 1:
        raise SystemExit("--points must be at least 1")
    if args.sample_stride < 1:
        raise SystemExit("--sample-stride must be at least 1")

    if not NUMBA_AVAILABLE:
        print("Warning: numba is not installed. Point-cloud generation will fall back to pure Python and be much slower.")

    start = time.perf_counter()
    positions, speeds = generate_aizawa_samples(args.points, args.burn_in, args.dt, args.sample_stride)
    normalized = normalize_points(positions)
    tone = compute_tone(normalized, speeds, args.color_mode)
    elapsed = time.perf_counter() - start
    print(f"Generated {args.points:,} Aizawa samples in {elapsed:.2f}s")
    return PointCloud(positions=normalized, tone=tone, speeds=speeds)


def perspective_matrix(fov_y_radians: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y_radians * 0.5)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def auto_rotation_matrix(elapsed: float) -> np.ndarray:
    yaw = elapsed * 0.23
    pitch = 0.62 + math.sin(elapsed * 0.31) * 0.18
    roll = math.sin(elapsed * 0.17) * 0.08
    return rotation_y(yaw) @ rotation_x(pitch) @ rotation_z(roll)


def compute_mvp(width: int, height: int, elapsed: float, camera_distance: float) -> np.ndarray:
    aspect = max(width / max(height, 1), 1e-6)
    projection = perspective_matrix(math.radians(48.0), aspect, 0.1, 100.0)
    view = translation_matrix(0.0, 0.0, -camera_distance)
    model = auto_rotation_matrix(elapsed)
    return projection @ view @ model


def palette_numpy(tone: np.ndarray) -> np.ndarray:
    tone = np.clip(tone.astype(np.float32, copy=False), 0.0, 1.0)
    low = np.array([0.08, 0.33, 0.98], dtype=np.float32)
    mid = np.array([0.95, 0.47, 0.15], dtype=np.float32)
    high = np.array([1.00, 0.95, 0.84], dtype=np.float32)

    first_mix = np.clip(tone / 0.58, 0.0, 1.0)
    first_mix = first_mix * first_mix * (3.0 - 2.0 * first_mix)
    ramp = (low * (1.0 - first_mix[:, None])) + (mid * first_mix[:, None])

    second_mix = np.clip((tone - 0.58) / 0.42, 0.0, 1.0)
    second_mix = second_mix * second_mix * (3.0 - 2.0 * second_mix)
    return (ramp * (1.0 - second_mix[:, None])) + (high * second_mix[:, None])


def project_to_ndc(positions: np.ndarray, mvp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.empty((positions.shape[0], 4), dtype=np.float32)
    homogeneous[:, :3] = positions
    homogeneous[:, 3] = 1.0

    clip = homogeneous @ mvp.T
    w = clip[:, 3]
    visible = w > 0.0
    ndc = clip[visible, :3] / w[visible, None]
    return ndc, visible


def ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def export_datashader_frame(
    path: Path,
    point_cloud: PointCloud,
    mvp: np.ndarray,
    width: int,
    height: int,
) -> None:
    try:
        import datashader as ds
        import pandas as pd
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Datashader export requires datashader, pandas, and Pillow. "
            "Install them before using --export-only or pressing S."
        ) from exc

    ndc, visible = project_to_ndc(point_cloud.positions, mvp)
    if ndc.size == 0:
        raise SystemExit("The current camera transform projected all points outside the view frustum.")

    tone = point_cloud.tone[visible].astype(np.float32, copy=False)
    visible_mask = (
        (ndc[:, 0] >= -1.05)
        & (ndc[:, 0] <= 1.05)
        & (ndc[:, 1] >= -1.05)
        & (ndc[:, 1] <= 1.05)
    )
    if not np.any(visible_mask):
        raise SystemExit("No points were visible inside the export bounds.")

    x = ndc[visible_mask, 0].astype(np.float32, copy=False)
    y = (-ndc[visible_mask, 1]).astype(np.float32, copy=False)
    tone = tone[visible_mask]

    dataframe = pd.DataFrame({"x": x, "y": y, "tone": tone})
    canvas = ds.Canvas(plot_width=width, plot_height=height, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))

    density = np.asarray(canvas.points(dataframe, "x", "y", agg=ds.count()), dtype=np.float32)
    tone_map = np.asarray(canvas.points(dataframe, "x", "y", agg=ds.mean("tone")), dtype=np.float32)

    density_log = np.log1p(density * 1.8)
    density_norm = density_log / max(float(np.max(density_log)), 1e-6)
    tone_map = np.nan_to_num(tone_map, nan=0.0)

    rgb = palette_numpy(tone_map.reshape(-1)).reshape(height, width, 3)
    rgb *= np.power(np.clip(density_norm, 0.0, 1.0), 0.78)[..., None]

    rng = np.random.default_rng(11)
    grain = (rng.random((height, width, 1), dtype=np.float32) - 0.5) * 0.12
    rgb = np.clip(rgb + grain * (0.15 + 0.85 * density_norm[..., None]), 0.0, 1.0)
    rgb = np.clip(rgb + np.power(density_norm, 2.0)[..., None] * 0.16, 0.0, 1.0)
    rgb = np.power(rgb, 0.92).astype(np.float32, copy=False)

    ensure_parent(path)
    Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB").save(path)
    print(f"Saved Datashader frame to {path}")


class ModernGLAizawaViewer:
    def __init__(self, point_cloud: PointCloud, args: argparse.Namespace) -> None:
        self.point_cloud = point_cloud
        self.args = args
        self.pygame, self.moderngl = self._import_runtime_stack()
        self.window_size = (args.width, args.height)
        self._setup_window()

        self.ctx = self.moderngl.create_context()
        self.ctx.enable(self.moderngl.BLEND)
        if hasattr(self.moderngl, "PROGRAM_POINT_SIZE"):
            self.ctx.enable(self.moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE
        self.ctx.viewport = (0, 0, *self.window_size)

        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.program["u_point_scale"].value = float(args.point_scale)

        vertex_data = np.empty((point_cloud.point_count, 4), dtype=np.float32)
        vertex_data[:, :3] = point_cloud.positions
        vertex_data[:, 3] = point_cloud.tone
        self.buffer = self.ctx.buffer(vertex_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.buffer, "3f 1f", "in_position", "in_tone")],
        )

        self.clock = self.pygame.time.Clock()
        self.start_time = time.perf_counter()
        self.last_export_time = 0.0

    def _import_runtime_stack(self):
        try:
            import moderngl
            import pygame
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "Real-time rendering requires pygame and moderngl. "
                "Install them before launching the viewer."
            ) from exc
        return pygame, moderngl

    def _setup_window(self) -> None:
        pygame = self.pygame
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        try:
            pygame.display.set_mode(self.window_size, flags, vsync=1)
        except TypeError:
            pygame.display.set_mode(self.window_size, flags)
        pygame.display.set_caption("Aizawa Strange Attractor | S export | Esc quit")

    def current_mvp(self) -> np.ndarray:
        elapsed = time.perf_counter() - self.start_time
        width, height = self.window_size
        return compute_mvp(width, height, elapsed, self.args.camera_distance)

    def save_current_frame(self) -> None:
        target = self.args.export
        export_datashader_frame(
            target,
            self.point_cloud,
            self.current_mvp(),
            self.args.export_width,
            self.args.export_height,
        )
        self.last_export_time = time.perf_counter()

    def run(self) -> None:
        pygame = self.pygame
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s and time.perf_counter() - self.last_export_time > 0.5:
                        self.save_current_frame()
                elif event.type in (pygame.WINDOWSIZECHANGED, pygame.VIDEORESIZE):
                    self.window_size = pygame.display.get_window_size()
                    self.ctx.viewport = (0, 0, *self.window_size)

            self.render_frame()
            pygame.display.flip()
            self.clock.tick(60)

            fps = self.clock.get_fps()
            if fps > 0.0:
                pygame.display.set_caption(
                    f"Aizawa Strange Attractor | {fps:5.1f} FPS | {self.point_cloud.point_count:,} points | S export"
                )

        self.close()

    def render_frame(self) -> None:
        mvp = self.current_mvp()
        self.program["u_mvp"].write(mvp.T.astype(np.float32, copy=False).tobytes())
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(mode=self.moderngl.POINTS)

    def close(self) -> None:
        self.vao.release()
        self.buffer.release()
        self.program.release()
        self.ctx.release()
        self.pygame.quit()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    point_cloud = build_point_cloud(args)

    if args.export_only:
        mvp = compute_mvp(args.export_width, args.export_height, elapsed=9.0, camera_distance=args.camera_distance)
        export_datashader_frame(args.export, point_cloud, mvp, args.export_width, args.export_height)
        return 0

    viewer = ModernGLAizawaViewer(point_cloud, args)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
