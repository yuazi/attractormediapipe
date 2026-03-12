from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class BackgroundLayers:
    texture_rgba: np.ndarray
    fog_rgba: np.ndarray


def _cache_key(width: int, height: int, accent: tuple[int, int, int]) -> tuple[int, int, tuple[int, int, int]]:
    return int(width), int(height), tuple(int(channel) for channel in accent)


def _seed_for(width: int, height: int, accent: tuple[int, int, int], salt: int) -> int:
    return (
        (width * 73856093)
        ^ (height * 19349663)
        ^ (accent[0] * 83492791)
        ^ (accent[1] * 2654435761)
        ^ (accent[2] * 97531)
        ^ (salt * 911382323)
    ) & 0xFFFFFFFF


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@lru_cache(maxsize=48)
def _texture_rgba(width: int, height: int, accent: tuple[int, int, int]) -> np.ndarray:
    accent_rgb = np.asarray(accent, dtype=np.float32) / np.float32(255.0)
    rng = np.random.default_rng(_seed_for(width, height, accent, 1))

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.array([1, 2, 4], dtype=np.uint8)
    rgba[:, :, 3] = 255

    center_x = width * 0.5
    cols = max(136, width // 11)
    rows = max(152, height // 7)
    base_point = np.array([138.0, 142.0, 149.0], dtype=np.float32) + accent_rgb * np.float32(10.0)
    glow_point = np.array([224.0, 220.0, 212.0], dtype=np.float32) + accent_rgb * np.float32(5.0)

    for row in range(rows):
        depth_ratio = row / max(1, rows - 1)
        base_y = height * (0.04 + depth_ratio**1.06 * 0.93)
        lateral_scale = width * (0.08 + depth_ratio**1.30 * 0.60)
        height_scale = height * (0.018 + depth_ratio * 0.072)
        world_z = depth_ratio * 8.0
        swirl = math.sin(world_z * 0.55 - 0.8) * width * 0.010
        for col in range(cols):
            u = col / max(1, cols - 1)
            world_x = (u - 0.5) * 5.2
            lattice_x = world_x + (rng.random() - 0.5) * (0.004 + depth_ratio * 0.010)

            noise_1 = math.sin(lattice_x * 1.82 + world_z * 0.92) * math.cos(lattice_x * 0.64 + world_z * 0.44)
            noise_2 = math.cos(lattice_x * 2.38 - world_z * 0.78) * math.sin(world_z * 1.22 + lattice_x * 0.36)
            noise_3 = math.sin((lattice_x * 0.42 + world_z * 0.30) * 5.2 + 0.6)
            noise_4 = math.cos((lattice_x * 0.96 - world_z * 0.68) * 4.4 - 0.3)
            organic_noise = (noise_1 + noise_2) * 0.76 + noise_3 * 0.28 + noise_4 * 0.14

            ridge = math.sin(lattice_x * 0.86 - world_z * 0.34 + 0.7) * 0.5 + 0.5
            ridge *= math.cos(lattice_x * 0.22 + world_z * 0.58 - 0.2) * 0.5 + 0.5
            surface = organic_noise * (0.42 + depth_ratio * 0.70) + ridge * (0.14 + depth_ratio * 0.20)

            vertical_ribbon = math.sin(lattice_x * 1.34 - world_z * 0.92 + 0.9) * height * 0.024
            suspended = max(0.0, ridge - 0.68) * (1.0 - depth_ratio) * height * 0.14

            sx = int(round(center_x + swirl + lattice_x * lateral_scale))
            sy = int(round(base_y - surface * height_scale - vertical_ribbon - suspended))
            if sx < 1 or sy < 1 or sx >= width - 1 or sy >= height - 1:
                continue

            density = min(0.995, 0.95 + ridge * 0.06 - (1.0 - depth_ratio) * 0.02)
            if rng.random() > density:
                continue

            center_glow = math.exp(-(((lattice_x * 0.52) ** 2) + (((depth_ratio - 0.76) / 0.28) ** 2)) * 2.0)
            intensity = np.clip(
                0.10 + depth_ratio * 0.38 + ridge * 0.26 + center_glow * 0.36,
                0.06,
                1.0,
            )
            alpha = int(np.clip(18.0 + intensity * 148.0, 16.0, 214.0))
            color = np.clip(
                base_point * (0.28 + intensity * 0.54) + glow_point * (center_glow * 0.24),
                0.0,
                255.0,
            ).astype(np.uint8)
            point_size = 2 if depth_ratio > 0.62 or intensity > 0.68 else 1
            x0 = max(0, sx - point_size // 2)
            x1 = min(width, x0 + point_size)
            y0 = max(0, sy - point_size // 2)
            y1 = min(height, y0 + point_size)
            patch = rgba[y0:y1, x0:x1]
            patch[:, :, :3] = np.maximum(patch[:, :, :3], color.reshape(1, 1, 3))
            patch[:, :, 3] = np.maximum(patch[:, :, 3], alpha)

    return _read_only(rgba)


@lru_cache(maxsize=48)
def _fog_rgba(width: int, height: int, accent: tuple[int, int, int]) -> np.ndarray:
    accent_rgb = np.asarray(accent, dtype=np.float32) / np.float32(255.0)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]

    terrain_haze = np.exp(-(((x * 1.1) ** 2) + (((y - 0.48) * 2.2) ** 2)) * 2.6)
    left_haze = np.exp(-((((x + 0.42) * 1.16) ** 2) + (((y - 0.36) * 1.9) ** 2)) * 3.2)
    right_haze = np.exp(-((((x - 0.36) * 1.08) ** 2) + (((y - 0.28) * 1.75) ** 2)) * 3.0)

    fog_color = np.clip(accent_rgb * np.float32(0.54) + np.array([0.09, 0.11, 0.13], dtype=np.float32), 0.0, 1.0)
    alpha = np.clip(terrain_haze * 30.0 + left_haze * 14.0 + right_haze * 12.0, 0.0, 44.0)

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.round(fog_color * 255.0).astype(np.uint8)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return _read_only(rgba)


def get_background_layers(width: int, height: int, accent: tuple[int, int, int]) -> BackgroundLayers:
    key = _cache_key(width, height, accent)
    return BackgroundLayers(texture_rgba=_texture_rgba(*key), fog_rgba=_fog_rgba(*key))


def texture_rgb(width: int, height: int, accent: tuple[int, int, int]) -> np.ndarray:
    return get_background_layers(width, height, accent).texture_rgba[:, :, :3].astype(np.float32) / np.float32(255.0)


def apply_fog_veil(
    rgb: np.ndarray,
    *,
    accent: tuple[int, int, int],
    fog_amount: float,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    fog_rgba = get_background_layers(width, height, accent).fog_rgba
    veil_rgb = fog_rgba[:, :, :3].astype(np.float32) / np.float32(255.0)
    veil_alpha = (fog_rgba[:, :, 3:4].astype(np.float32) / np.float32(255.0)) * np.float32(
        max(0.0, min(1.0, fog_amount))
    )
    return rgb * (1.0 - veil_alpha) + veil_rgb * veil_alpha


def compose_textured_density(
    density_rgb: np.ndarray,
    *,
    accent: tuple[int, int, int],
    fog_amount: float,
) -> np.ndarray:
    height, width = density_rgb.shape[:2]
    background = texture_rgb(width, height, accent)
    fogged = apply_fog_veil(background, accent=accent, fog_amount=fog_amount)
    return np.clip(fogged + density_rgb.astype(np.float32, copy=False), 0.0, 1.0)
