from __future__ import annotations

import colorsys
import math
from typing import Dict, Tuple

import pygame

from config import (
    GLOW_SPRITE_SIZE,
    PARTICLE_ALPHA_MAX,
    PARTICLE_ALPHA_MIN,
    PARTICLE_SIZE_MAX,
    PARTICLE_SIZE_MIN,
)


HALO_STRIDE = 7
MIDPOINT_STRIDE = 3
SPINE_SEGMENTS = 120


def _quantize_channel(value: int, step: int = 12) -> int:
    return max(0, min(255, int(round(value / step) * step)))


class ParticleRenderer:
    def __init__(self) -> None:
        self._masters = {
            "spark": self._make_glow_master(GLOW_SPRITE_SIZE, exponent=6.5, core_ratio=0.18),
            "halo": self._make_glow_master(GLOW_SPRITE_SIZE, exponent=2.3, core_ratio=0.04),
            "dust": self._make_glow_master(GLOW_SPRITE_SIZE, exponent=9.0, core_ratio=0.10),
        }
        self._sprite_cache: Dict[Tuple[str, int, Tuple[int, int, int], int], pygame.Surface] = {}

    def _make_glow_master(self, size: int, *, exponent: float, core_ratio: float) -> pygame.Surface:
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        cx = cy = size // 2
        radius = size / 2
        for x in range(size):
            for y in range(size):
                distance = math.hypot(x - cx, y - cy) / radius
                if distance >= 1.0:
                    alpha = 0
                elif distance <= core_ratio:
                    alpha = 255
                else:
                    normalized = 1.0 - (distance - core_ratio) / max(1e-6, 1.0 - core_ratio)
                    alpha = int(255 * max(0.0, normalized) ** exponent)
                surface.set_at((x, y), (255, 255, 255, alpha))
        return surface

    def _sprite(self, kind: str, size: int, color: Tuple[int, int, int], alpha: int) -> pygame.Surface:
        quantized_color = tuple(_quantize_channel(channel) for channel in color)
        alpha_bucket = int(round(alpha / 8) * 8)
        key = (kind, size, quantized_color, alpha_bucket)
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached

        sprite = pygame.transform.smoothscale(self._masters[kind], (size, size))
        sprite = sprite.copy()
        sprite.fill((*quantized_color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        sprite.set_alpha(alpha_bucket)
        self._sprite_cache[key] = sprite
        return sprite

    def _mix_color(self, color: Tuple[int, int, int], target: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
        blend = max(0.0, min(1.0, amount))
        return tuple(
            int(round(channel * (1.0 - blend) + target_channel * blend))
            for channel, target_channel in zip(color, target)
        )

    def _volume_offsets(self, points_2d, index: int, radius: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        previous_point = points_2d[index - 1] if index > 0 else points_2d[index]
        next_point = points_2d[index + 1] if index + 1 < len(points_2d) else points_2d[index]
        tx = float(next_point[0] - previous_point[0])
        ty = float(next_point[1] - previous_point[1])
        length = max(1e-5, math.hypot(tx, ty))
        tangent = (tx / length, ty / length)
        normal = (-tangent[1], tangent[0])
        spread = 0.55 + 0.45 * abs(math.sin(index * 0.41))
        drift = math.cos(index * 0.19)
        primary = (
            normal[0] * radius * spread + tangent[0] * radius * 0.22 * drift,
            normal[1] * radius * spread + tangent[1] * radius * 0.22 * drift,
        )
        secondary = (
            -normal[0] * radius * (0.3 + 0.4 * abs(drift)) - tangent[0] * radius * 0.16 * spread,
            -normal[1] * radius * (0.3 + 0.4 * abs(drift)) - tangent[1] * radius * 0.16 * spread,
        )
        return primary, secondary

    def _draw_sprite(
        self,
        surface: pygame.Surface,
        kind: str,
        point,
        size: int,
        color: Tuple[int, int, int],
        alpha: int,
        *,
        additive: bool = False,
    ) -> None:
        if size <= 0 or alpha <= 0:
            return
        sprite = self._sprite(kind, size, color, alpha)
        destination = (int(round(point[0])) - size // 2, int(round(point[1])) - size // 2)
        if additive:
            surface.blit(sprite, destination, special_flags=pygame.BLEND_RGBA_ADD)
        else:
            surface.blit(sprite, destination)

    def _draw_spine(
        self,
        surface: pygame.Surface,
        points_2d,
        depths,
        color: Tuple[int, int, int],
        glow_gain: float,
    ) -> None:
        if len(points_2d) < 2:
            return

        recent_segments = min(len(points_2d) - 1, SPINE_SEGMENTS)
        recent_points = points_2d[-(recent_segments + 1):]
        recent_depths = depths[-(recent_segments + 1):]
        spine_color = self._mix_color(color, (255, 255, 255), 0.16)
        total = max(1, len(recent_points) - 1)
        for idx in range(1, len(recent_points)):
            age = idx / total
            depth = float(recent_depths[idx])
            alpha = int((8 + 46 * age) * (0.5 + 0.26 * glow_gain) * (0.42 + 0.58 * depth))
            width = 1
            start = (int(round(recent_points[idx - 1][0])), int(round(recent_points[idx - 1][1])))
            end = (int(round(recent_points[idx][0])), int(round(recent_points[idx][1])))
            pygame.draw.line(surface, (*spine_color, min(255, alpha)), start, end, width)

        hotspot_size = PARTICLE_SIZE_MAX + 1
        hotspot_alpha = int(42 * (0.5 + 0.24 * glow_gain))
        self._draw_sprite(surface, "halo", recent_points[-1], hotspot_size * 2, spine_color, hotspot_alpha, additive=True)
        self._draw_sprite(surface, "spark", recent_points[-1], hotspot_size, self._mix_color(spine_color, (255, 255, 255), 0.08), 84)

    def draw(
        self,
        surface: pygame.Surface,
        points_2d,
        depths,
        canonical_color: Tuple[int, int, int],
        luminosity: float,
        particle_count: int,
    ) -> None:
        if len(points_2d) == 0:
            return

        base_h, base_s, base_v = colorsys.rgb_to_hsv(
            canonical_color[0] / 255.0,
            canonical_color[1] / 255.0,
            canonical_color[2] / 255.0,
        )
        stream_count = max(1, min(10, int(round(particle_count))))
        glow_gain = max(0.05, min(1.0, luminosity))

        for stream_index in range(stream_count):
            stream_points = points_2d[stream_index::stream_count]
            stream_depths = depths[stream_index::stream_count]
            if len(stream_points) == 0:
                continue
            self._draw_stream(
                surface,
                stream_points,
                stream_depths,
                base_h,
                base_s,
                base_v,
                glow_gain,
            )

    def _draw_stream(
        self,
        surface: pygame.Surface,
        points_2d,
        depths,
        base_h: float,
        base_s: float,
        base_v: float,
        glow_gain: float,
    ) -> None:
        total = max(1, len(points_2d) - 1)
        halo_stride = HALO_STRIDE if len(points_2d) <= 3200 else HALO_STRIDE + 1

        for idx, point in enumerate(points_2d):
            age = idx / total
            depth = float(depths[idx])
            focus = age**1.35
            sample_gate = 3 if age < 0.38 else 2 if age < 0.92 else 1
            draw_primary = (idx % sample_gate) == 0
            size = int(round(PARTICLE_SIZE_MIN + focus * (PARTICLE_SIZE_MAX - PARTICLE_SIZE_MIN) + depth * 0.75))
            halo_size = int(round(size * (1.55 + focus * 0.4)))
            base_alpha = PARTICLE_ALPHA_MIN + focus * (PARTICLE_ALPHA_MAX - PARTICLE_ALPHA_MIN)
            alpha = int(base_alpha * (0.2 + 0.46 * glow_gain) * (0.42 + 0.58 * depth))

            particle_h = base_h
            particle_s = min(1.0, max(0.22, base_s * (0.88 + 0.12 * depth)))
            particle_v = min(1.0, max(0.12, base_v * (0.18 + age * 0.34 + glow_gain * 0.9)))

            halo_rgb = tuple(
                int(channel * 255)
                for channel in colorsys.hsv_to_rgb(particle_h, particle_s, min(1.0, particle_v * (0.78 + 0.18 * depth)))
            )
            spark_rgb = self._mix_color(
                tuple(
                    int(channel * 255)
                    for channel in colorsys.hsv_to_rgb(particle_h, particle_s * 0.66, min(1.0, 0.62 + particle_v * 0.56))
                ),
                (255, 255, 255),
                0.02 + 0.05 * focus,
            )

            if draw_primary and (idx % halo_stride == 0 or focus > 0.82):
                self._draw_sprite(surface, "halo", point, halo_size, halo_rgb, int(alpha * (0.06 + 0.06 * glow_gain)), additive=True)

            if draw_primary:
                self._draw_sprite(surface, "spark", point, size, spark_rgb, alpha)

            if idx % MIDPOINT_STRIDE == 0 and age > 0.12:
                primary_offset, secondary_offset = self._volume_offsets(points_2d, idx, size * (1.2 + focus * 2.2))
                dust_rgb = self._mix_color(halo_rgb, (255, 255, 255), 0.06)
                dust_alpha = int(alpha * (0.16 + 0.1 * age))
                dust_size = max(1, size - 1)
                self._draw_sprite(surface, "dust", (point[0] + primary_offset[0], point[1] + primary_offset[1]), dust_size, dust_rgb, dust_alpha)
                if idx % (MIDPOINT_STRIDE * 2) == 0:
                    self._draw_sprite(
                        surface,
                        "dust",
                        (point[0] + secondary_offset[0], point[1] + secondary_offset[1]),
                        max(1, dust_size - 1),
                        dust_rgb,
                        int(dust_alpha * 0.82),
                    )

        self._draw_spine(
            surface,
            points_2d,
            depths,
            tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(base_h, min(1.0, base_s * 0.76), 1.0)),
            glow_gain,
        )
