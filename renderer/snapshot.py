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

from attractors.manager import create_active_attractor, normalize_points
from config import (
    DEFAULT_DT,
    SCREENSHOT_DIR,
    SCREENSHOT_PREFIX,
    SNAPSHOT_BURN_IN,
    SNAPSHOT_HEIGHT,
    SNAPSHOT_SAMPLE_STRIDE,
    SNAPSHOT_SAMPLES,
    SNAPSHOT_WIDTH,
)

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


def inferno_palette(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0.001, 0.000, 0.014],
            [0.208, 0.016, 0.350],
            [0.562, 0.141, 0.410],
            [0.865, 0.317, 0.226],
            [0.987, 0.645, 0.039],
            [0.988, 0.998, 0.645],
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, num=len(stops), dtype=np.float32)
    clipped = np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)
    channels = [np.interp(clipped, positions, stops[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def _render_density_image(density: np.ndarray, luminosity: float) -> np.ndarray:
    density = np.asarray(density, dtype=np.float32)
    density = np.flipud(density)
    density_log = np.log1p(density * (1.6 + max(0.0, luminosity)))
    peak = max(float(np.max(density_log)), 1e-6)
    density_norm = np.clip(density_log / peak, 0.0, 1.0)

    rgb = inferno_palette(density_norm.reshape(-1)).reshape(density.shape[0], density.shape[1], 3)
    rgb *= np.power(density_norm[..., None], 0.78)
    rgb += np.power(density_norm[..., None], 2.2) * (0.10 + 0.25 * max(0.05, luminosity))
    return np.clip(np.power(rgb, 0.92), 0.0, 1.0)


def export_attractor_snapshot(request: SnapshotRequest) -> str:
    ensure_snapshot_environment()
    ds, pd = _import_datashader_dependencies()

    attractor = create_active_attractor(request.attractor_name)
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

    x = ndc[in_bounds, 0].astype(np.float32, copy=False)
    y = (-ndc[in_bounds, 1]).astype(np.float32, copy=False)
    dataframe = pd.DataFrame({"x": x, "y": y})
    canvas = ds.Canvas(plot_width=request.width, plot_height=request.height, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    density = canvas.points(dataframe, "x", "y", agg=ds.count())
    rgb = _render_density_image(np.asarray(density), request.luminosity)

    output = request.output_path or snapshot_filename()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB").save(output_path)
    return str(output_path)


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
            path = export_attractor_snapshot(request)
        except Exception as exc:  # pragma: no cover - exercised via integration path
            message = f"Snapshot failed: {exc}"
        else:
            message = f"Snapshot saved: {Path(path).name}"

        with self._lock:
            self._is_running = False
            self._message = message
            self._thread = None
