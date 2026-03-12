from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from attractors.manager import HUD_ACCENT_OVERRIDES, create_active_attractor, normalize_points
from config import (
    DEFAULT_DT,
    DEFAULT_FOG,
    SCREENSHOT_DIR,
    SCREENSHOT_PREFIX,
    SNAPSHOT_BURN_IN,
    SNAPSHOT_HEIGHT,
    SNAPSHOT_SAMPLE_STRIDE,
    SNAPSHOT_SAMPLES,
    SNAPSHOT_WIDTH,
)

from .background import compose_textured_density
from .common import animated_points, compute_mvp, project_ndc


@dataclass(frozen=True)
class SnapshotRequest:
    attractor_name: str
    yaw: float
    pitch: float
    roll: float
    zoom: float
    time_value: float
    luminosity: float
    fog: float = DEFAULT_FOG
    output_path: str = ""
    width: int = SNAPSHOT_WIDTH
    height: int = SNAPSHOT_HEIGHT
    sample_count: int = SNAPSHOT_SAMPLES
    burn_in: int = SNAPSHOT_BURN_IN
    sample_stride: int = SNAPSHOT_SAMPLE_STRIDE
    dt: float = DEFAULT_DT
    state: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("snapshot dimensions must be >= 1")
        if self.sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if not 0.0 <= self.fog <= 1.0:
            raise ValueError("fog must be within [0.0, 1.0]")


@dataclass(frozen=True)
class SnapshotExportResult:
    clean_path: str
    textured_path: str


@dataclass(frozen=True)
class SnapshotOutputPaths:
    clean: Path
    textured: Path


def ensure_snapshot_environment() -> str:
    cache_dir = os.path.join(tempfile.gettempdir(), "attractor-numba-cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", cache_dir)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "attractor-mpl-cache"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    return cache_dir


def _clear_partial_datashader_modules() -> None:
    for name in list(sys.modules):
        if name == "datashader" or name.startswith("datashader."):
            sys.modules.pop(name, None)


@contextmanager
def _datashader_numba_cache_disabled():
    import numba

    original_jit = numba.jit
    original_njit = numba.njit

    def _without_cache(factory):
        def wrapper(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["cache"] = False
            return factory(*args, **kwargs)

        return wrapper

    numba.jit = _without_cache(original_jit)
    numba.njit = _without_cache(original_njit)
    try:
        yield
    finally:
        numba.jit = original_jit
        numba.njit = original_njit


def _import_datashader_dependencies():
    try:
        import datashader as ds
        import pandas as pd
    except RuntimeError as exc:
        if "cannot cache function" not in str(exc):
            raise
        _clear_partial_datashader_modules()
        with _datashader_numba_cache_disabled():
            import datashader as ds  # type: ignore[no-redef]
            import pandas as pd  # type: ignore[no-redef]
    except ImportError as exc:
        raise RuntimeError("Datashader snapshots require datashader and pandas to be installed") from exc
    return ds, pd


def snapshot_filename(prefix: str = SCREENSHOT_PREFIX) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path(SCREENSHOT_DIR) / f"{prefix}_{timestamp}.png")


def snapshot_output_paths(output_path: str = "", prefix: str = SCREENSHOT_PREFIX) -> SnapshotOutputPaths:
    base_path = Path(output_path) if output_path else Path(snapshot_filename(prefix))
    suffix = base_path.suffix or ".png"
    stem = base_path.stem or prefix
    for variant in ("_clean", "_textured"):
        if stem.endswith(variant):
            stem = stem[: -len(variant)] or prefix
            break

    return SnapshotOutputPaths(
        clean=base_path.with_name(f"{stem}_clean{suffix}"),
        textured=base_path.with_name(f"{stem}_textured{suffix}"),
    )


def _accent_palette(values: np.ndarray, accent: tuple[int, int, int]) -> np.ndarray:
    accent_rgb = np.asarray(accent, dtype=np.float32) / np.float32(255.0)
    shadow = accent_rgb * np.float32(0.12)
    low = accent_rgb * np.float32(0.34)
    mid = accent_rgb * np.float32(0.60)
    bright = accent_rgb + (1.0 - accent_rgb) * np.float32(0.18)
    glow = accent_rgb + (1.0 - accent_rgb) * np.float32(0.42)
    hot = accent_rgb + (1.0 - accent_rgb) * np.float32(0.82)
    stops = np.stack(
        (
            np.zeros(3, dtype=np.float32),
            shadow,
            low,
            mid,
            bright,
            glow,
            hot,
        ),
        axis=0,
    )
    positions = np.linspace(0.0, 1.0, num=len(stops), dtype=np.float32)
    clipped = np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)
    channels = [np.interp(clipped, positions, stops[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def _render_density_image(density: np.ndarray, luminosity: float, accent: tuple[int, int, int]) -> np.ndarray:
    density = np.asarray(density, dtype=np.float32)
    density = np.flipud(density)
    density_log = np.log1p(density * (1.6 + max(0.0, luminosity)))
    peak = max(float(np.max(density_log)), 1e-6)
    density_norm = np.clip(density_log / peak, 0.0, 1.0)

    accent_rgb = (np.asarray(accent, dtype=np.float32) / np.float32(255.0)).reshape(1, 1, 3)
    rgb = _accent_palette(density_norm.reshape(-1), accent).reshape(density.shape[0], density.shape[1], 3)
    rgb *= np.power(density_norm[..., None], 0.78)
    rgb += accent_rgb * np.power(density_norm[..., None], 1.35) * np.float32(0.08)
    rgb += np.power(density_norm[..., None], 2.2) * (0.08 + 0.22 * max(0.05, luminosity))
    return np.clip(np.power(rgb, 0.92), 0.0, 1.0)


def _cover_frame_points(points_2d: np.ndarray, overscan: float = 1.02) -> np.ndarray:
    if len(points_2d) == 0:
        return np.empty((0, 2), dtype=np.float32)

    coords = np.asarray(points_2d, dtype=np.float32)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) * np.float32(0.5)
    centered = coords - center
    half_extents = np.max(np.abs(centered), axis=0)
    max_extent = float(np.max(half_extents))
    min_extent = float(np.min(half_extents))
    if max_extent < 1e-6:
        return centered.astype(np.float32, copy=False)
    cover_extent = min_extent if min_extent >= 1e-6 else max_extent
    scale = np.float32(overscan / cover_extent)
    return (centered * scale).astype(np.float32, copy=False)


def export_attractor_snapshot(request: SnapshotRequest) -> SnapshotExportResult:
    ensure_snapshot_environment()
    ds, pd = _import_datashader_dependencies()

    attractor = create_active_attractor(request.attractor_name)
    accent = HUD_ACCENT_OVERRIDES.get(attractor.name, attractor.color)
    if request.state is not None:
        attractor.set_state(request.state)

    points = attractor.sample_points(
        request.sample_count,
        dt=request.dt,
        burn_in=request.burn_in,
        sample_stride=request.sample_stride,
    )
    normalized = normalize_points(points) * np.float32(attractor.scale_hint)
    animated = animated_points(normalized, request.time_value)
    mvp = compute_mvp(
        request.width,
        request.height,
        yaw=request.yaw,
        pitch=request.pitch,
        roll=request.roll,
        zoom=request.zoom,
    )
    ndc, _ = project_ndc(animated, mvp)
    if ndc.size == 0:
        raise RuntimeError("Snapshot export projected all points outside the view frustum")

    in_bounds = (
        (ndc[:, 0] >= -1.05)
        & (ndc[:, 0] <= 1.05)
        & (ndc[:, 1] >= -1.05)
        & (ndc[:, 1] <= 1.05)
    )
    if not np.any(in_bounds):
        raise RuntimeError("Snapshot export produced no visible points in bounds")

    framed = _cover_frame_points(ndc[in_bounds, :2])
    reframed_in_bounds = (
        (framed[:, 0] >= -1.0)
        & (framed[:, 0] <= 1.0)
        & (framed[:, 1] >= -1.0)
        & (framed[:, 1] <= 1.0)
    )
    if not np.any(reframed_in_bounds):
        raise RuntimeError("Snapshot export reframing removed all visible points")

    x = framed[reframed_in_bounds, 0].astype(np.float32, copy=False)
    y = (-framed[reframed_in_bounds, 1]).astype(np.float32, copy=False)
    dataframe = pd.DataFrame({"x": x, "y": y})
    canvas = ds.Canvas(plot_width=request.width, plot_height=request.height, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    density = canvas.points(dataframe, "x", "y", agg=ds.count())
    clean_rgb = _render_density_image(np.asarray(density), request.luminosity, accent)
    textured_rgb = compose_textured_density(clean_rgb, accent=accent, fog_amount=request.fog)

    output_paths = snapshot_output_paths(request.output_path)
    output_paths.clean.parent.mkdir(parents=True, exist_ok=True)
    output_paths.textured.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((clean_rgb * 255.0).astype(np.uint8), mode="RGB").save(output_paths.clean)
    Image.fromarray((textured_rgb * 255.0).astype(np.uint8), mode="RGB").save(output_paths.textured)
    return SnapshotExportResult(clean_path=str(output_paths.clean), textured_path=str(output_paths.textured))


class SnapshotController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._is_running = False
        self._message = ""

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    def start(self, request: SnapshotRequest) -> bool:
        with self._lock:
            if self._is_running:
                return False
            self._is_running = True
            self._message = "Exporting snapshot..."
        self._thread = threading.Thread(target=self._run, args=(request,), name="snapshot-export", daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join()

    def _run(self, request: SnapshotRequest) -> None:
        try:
            result = export_attractor_snapshot(request)
        except Exception as exc:  # pragma: no cover - exercised via integration path
            message = f"Snapshot failed: {exc}"
        else:
            message = f"Snapshots saved: {Path(result.clean_path).name}, {Path(result.textured_path).name}"

        with self._lock:
            self._is_running = False
            self._message = message
            self._thread = None
