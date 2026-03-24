from __future__ import annotations

from dataclasses import dataclass


PaletteStop = tuple[float, tuple[int, int, int]]


@dataclass(frozen=True)
class SnapshotPreset:
    name: str
    palette_name: str
    palette: tuple[PaletteStop, ...]
    background: tuple[int, int, int]
    texture_mode: str
    mood: str


SNAPSHOT_PRESETS: dict[str, SnapshotPreset] = {
    "nebula": SnapshotPreset(
        name="nebula",
        palette_name="fire",
        palette=(
            (0.00, (6, 4, 10)),
            (0.18, (56, 11, 27)),
            (0.42, (168, 36, 38)),
            (0.70, (244, 120, 44)),
            (1.00, (255, 236, 181)),
        ),
        background=(5, 4, 8),
        texture_mode="grain_scanlines",
        mood="warm",
    ),
    "blueprint": SnapshotPreset(
        name="blueprint",
        palette_name="single-hue blue",
        palette=(
            (0.00, (8, 19, 46)),
            (0.26, (19, 54, 110)),
            (0.62, (74, 146, 232)),
            (1.00, (220, 241, 255)),
        ),
        background=(7, 19, 44),
        texture_mode="none",
        mood="technical",
    ),
    "void": SnapshotPreset(
        name="void",
        palette_name="viridis",
        palette=(
            (0.00, (0, 0, 0)),
            (0.18, (68, 1, 84)),
            (0.45, (49, 104, 142)),
            (0.72, (53, 183, 121)),
            (1.00, (253, 231, 37)),
        ),
        background=(0, 0, 0),
        texture_mode="vignette",
        mood="minimal",
    ),
    "print": SnapshotPreset(
        name="print",
        palette_name="greyscale",
        palette=(
            (0.00, (255, 255, 255)),
            (0.20, (224, 224, 224)),
            (0.55, (150, 150, 150)),
            (1.00, (12, 12, 12)),
        ),
        background=(255, 255, 255),
        texture_mode="halftone",
        mood="printable",
    ),
}


def available_snapshot_presets() -> tuple[str, ...]:
    return tuple(SNAPSHOT_PRESETS)


def get_snapshot_preset(name: str) -> SnapshotPreset:
    normalized = name.strip().lower()
    try:
        return SNAPSHOT_PRESETS[normalized]
    except KeyError as exc:
        available = ", ".join(available_snapshot_presets())
        raise KeyError(f"Unknown snapshot preset '{name}'. Available: {available}") from exc


def cycle_snapshot_preset(current_name: str, step: int = 1) -> str:
    names = available_snapshot_presets()
    if not names:
        raise ValueError("No snapshot presets have been configured")
    normalized = current_name.strip().lower()
    try:
        index = names.index(normalized)
    except ValueError:
        index = 0
    return names[(index + step) % len(names)]
