from __future__ import annotations

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


MIDPOINT_STRIDE = 2
SPINE_SEGMENTS = 40
SEGMENT_FILL_THRESHOLD = 1.2
SEGMENT_FILL_SPACING = 0.75
SEGMENT_FILL_MAX_STEPS = 8


def _quantize_channel(value: int, step: int = 12) -> int:
    return max(0, min(255, int(round(value / step) * step)))


class ParticleRenderer:
    def __init__(self) -> None:
        self._masters = {
            "spark": self._make_glow_master(GLOW_SPRITE_SIZE, exponent=8.2, core_ratio=0.08),
            "dust": self._make_glow_master(GLOW_SPRITE_SIZE, exponent=12.0, core_ratio=0.0),
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
        alpha_bucket = max(0, min(255, int(round(alpha / 8) * 8)))
        key = (kind, size, quantized_color, alpha_bucket)
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached

        if size == 1:
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            sprite.fill((*quantized_color, alpha_bucket))
        elif size <= 3:
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(
                sprite,
                (*quantized_color, alpha_bucket),
                (size // 2, size // 2),
                max(1, size // 2),
            )
        else:
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

    def _boost_color(self, color: Tuple[int, int, int], floor: int) -> Tuple[int, int, int]:
        return tuple(max(floor, channel) for channel in color)

    def _trail_frame(self, points_2d, index: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        previous_point = points_2d[index - 1] if index > 0 else points_2d[index]
        next_point = points_2d[index + 1] if index + 1 < len(points_2d) else points_2d[index]
        tx = float(next_point[0] - previous_point[0])
        ty = float(next_point[1] - previous_point[1])
        length = max(1e-5, math.hypot(tx, ty))
        tangent = (tx / length, ty / length)
        normal = (-tangent[1], tangent[0])
        return tangent, normal

    def _noise(self, seed: float) -> float:
        value = math.sin(seed * 12.9898 + 78.233) * 43758.5453
        return value - math.floor(value)

    def _volume_offsets(self, points_2d, index: int, radius: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        tangent, normal = self._trail_frame(points_2d, index)
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

    def _grain_offsets(self, points_2d, index: int, radius: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        tangent, normal = self._trail_frame(points_2d, index)
        lateral = (self._noise(index * 0.73 + 1.1) - 0.5) * radius * 2.2
        axial = (self._noise(index * 1.17 + 4.7) - 0.5) * radius * 0.9
        mirrored_lateral = (self._noise(index * 0.49 + 8.3) - 0.5) * radius * 1.8
        mirrored_axial = (self._noise(index * 0.91 + 2.4) - 0.5) * radius * 0.7
        primary = (
            normal[0] * lateral + tangent[0] * axial,
            normal[1] * lateral + tangent[1] * axial,
        )
        secondary = (
            -normal[0] * mirrored_lateral + tangent[0] * mirrored_axial,
            -normal[1] * mirrored_lateral + tangent[1] * mirrored_axial,
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
        spine_color = self._mix_color(color, (255, 255, 255), 0.96)
        total = max(1, len(recent_points))
        for idx in range(0, len(recent_points), 2):
            age = idx / total
            depth = float(recent_depths[idx])
            alpha = int((8 + 18 * age) * (0.48 + 0.18 * glow_gain) * (0.45 + 0.55 * depth))
            self._draw_sprite(surface, "spark", recent_points[idx], 1, spine_color, alpha)

        self._draw_sprite(surface, "spark", recent_points[-1], 1, (255, 255, 255), int(20 + 16 * glow_gain))

    def _draw_segment_fill(
        self,
        surface: pygame.Surface,
        start_point,
        end_point,
        color: Tuple[int, int, int],
        alpha: int,
    ) -> None:
        dx = float(end_point[0] - start_point[0])
        dy = float(end_point[1] - start_point[1])
        distance = math.hypot(dx, dy)
        if distance <= SEGMENT_FILL_THRESHOLD:
            return

        pygame.draw.line(
            surface,
            (*color, max(0, min(255, int(alpha * 0.24)))),
            (int(round(start_point[0])), int(round(start_point[1]))),
            (int(round(end_point[0])), int(round(end_point[1]))),
            1,
        )

        step_count = min(SEGMENT_FILL_MAX_STEPS, max(0, int(math.ceil(distance / SEGMENT_FILL_SPACING)) - 1))
        if step_count <= 0:
            return

        for step in range(1, step_count + 1):
            t = step / (step_count + 1)
            sample_point = (start_point[0] + dx * t, start_point[1] + dy * t)
            self._draw_sprite(surface, "spark", sample_point, 1, color, alpha)

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

        stream_count = max(1, min(10, int(round(particle_count))))
        glow_gain = max(0.05, min(1.0, luminosity))
        grain_color = self._boost_color(canonical_color, 84)
        tint_color = self._mix_color(grain_color, (255, 255, 255), 0.22)
        accent_color = self._mix_color(grain_color, (255, 255, 255), 0.45)

        for stream_index in range(stream_count):
            stream_points = points_2d[stream_index::stream_count]
            stream_depths = depths[stream_index::stream_count]
            if len(stream_points) == 0:
                continue
            detail_stride = 1 if stream_count <= 3 else 2 if stream_count <= 6 else 3
            echo_strength = 1.0 if stream_count <= 3 else 0.5 if stream_count <= 6 else 0.0
            self._draw_stream(
                surface,
                stream_points,
                stream_depths,
                grain_color,
                tint_color,
                accent_color,
                glow_gain,
                detail_stride,
                echo_strength,
            )

    def _draw_stream(
        self,
        surface: pygame.Surface,
        points_2d,
        depths,
        grain_color: Tuple[int, int, int],
        tint_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int],
        glow_gain: float,
        detail_stride: int,
        echo_strength: float,
    ) -> None:
        total = max(1, len(points_2d) - 1)
        for idx, point in enumerate(points_2d):
            age = idx / total
            depth = float(depths[idx])
            age_gain = 0.34 + 0.66 * age**0.82
            size = 1
            base_alpha = PARTICLE_ALPHA_MIN + age_gain * (PARTICLE_ALPHA_MAX - PARTICLE_ALPHA_MIN) * 0.96
            alpha = int(base_alpha * (0.8 + 0.42 * glow_gain) * (0.68 + 0.42 * depth))
            spark_color = accent_color if age > 0.84 else grain_color

            self._draw_sprite(surface, "spark", point, size, spark_color, alpha)
            if idx > 0:
                self._draw_segment_fill(surface, points_2d[idx - 1], point, spark_color, alpha)

            if idx % (MIDPOINT_STRIDE * detail_stride) == 0:
                grain_radius = 0.8 + depth * 0.7 + age_gain * 1.8
                primary_offset, secondary_offset = self._grain_offsets(points_2d, idx, grain_radius)
                dust_alpha = int(alpha * (0.32 + 0.1 * age_gain))
                self._draw_sprite(
                    surface,
                    "dust",
                    (point[0] + primary_offset[0], point[1] + primary_offset[1]),
                    1,
                    tint_color,
                    dust_alpha,
                )
                if age > 0.08 and echo_strength > 0.0:
                    self._draw_sprite(
                        surface,
                        "dust",
                        (point[0] + secondary_offset[0], point[1] + secondary_offset[1]),
                        1,
                        grain_color,
                        int(dust_alpha * (0.5 + 0.32 * echo_strength)),
                    )
                if idx % (MIDPOINT_STRIDE * 3 * detail_stride) == 0 and echo_strength > 0.0:
                    volume_offset, echo_offset = self._volume_offsets(points_2d, idx, grain_radius * 0.8)
                    self._draw_sprite(
                        surface,
                        "dust",
                        (point[0] + volume_offset[0], point[1] + volume_offset[1]),
                        1,
                        tint_color,
                        int(dust_alpha * (0.18 + 0.14 * echo_strength)),
                    )
                    self._draw_sprite(
                        surface,
                        "dust",
                        (point[0] + echo_offset[0], point[1] + echo_offset[1]),
                        1,
                        accent_color,
                        int(dust_alpha * (0.12 + 0.12 * echo_strength)),
                    )

        self._draw_spine(
            surface,
            points_2d,
            depths,
            tint_color,
            glow_gain,
        )
