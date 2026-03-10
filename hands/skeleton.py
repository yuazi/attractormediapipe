from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple

import pygame
from PIL import Image, ImageDraw, ImageFont

from config import SKELETON_COLOR


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_TEXT_CACHE: dict[tuple[str, Tuple[int, int, int], int], pygame.Surface] = {}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cached = _FONT_CACHE.get(size)
    if cached is None:
        try:
            cached = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=size)
        except OSError:
            cached = ImageFont.load_default(size=size)
        _FONT_CACHE[size] = cached
    return cached


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _render_text(text: str, color: Tuple[int, int, int], size: int) -> pygame.Surface:
    key = (text, color, size)
    cached = _TEXT_CACHE.get(key)
    if cached is not None:
        return cached

    font = _font(size)
    bbox = font.getbbox(text or " ")
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((1 - bbox[0], 1 - bbox[1]), text, font=font, fill=(*color, 255))
    rendered = pygame.image.fromstring(image.tobytes(), image.size, image.mode).convert_alpha()
    _TEXT_CACHE[key] = rendered
    return rendered


def draw_hand_skeleton(
    surface: pygame.Surface,
    landmarks: Sequence[Tuple[float, float, float]],
    width: int,
    height: int,
    color: Tuple[int, int, int] = SKELETON_COLOR,
    caption: Optional[str] = None,
    connections: Iterable[Tuple[int, int]] = HAND_CONNECTIONS,
) -> None:
    del connections
    if len(landmarks) < 9:
        return

    scale = max(1.0, min(width / 320.0, height / 180.0))
    line_width = max(2, int(round(4 * scale)))
    point_radius = max(4, int(round(6 * scale)))
    shadow_radius = point_radius + max(2, int(round(4 * scale)))
    font_size = max(15, int(round(21 * scale)))
    text_gap = max(8, int(round(12 * scale)))

    thumb_tip = (int(landmarks[4][0] * width), int(landmarks[4][1] * height))
    index_tip = (int(landmarks[8][0] * width), int(landmarks[8][1] * height))
    glow_color = (*color, 70)
    pygame.draw.line(surface, glow_color, thumb_tip, index_tip, line_width + max(2, int(round(5 * scale))))
    pygame.draw.line(surface, color, thumb_tip, index_tip, line_width)
    pygame.draw.circle(surface, glow_color, thumb_tip, shadow_radius)
    pygame.draw.circle(surface, glow_color, index_tip, shadow_radius)
    pygame.draw.circle(surface, color, thumb_tip, point_radius)
    pygame.draw.circle(surface, color, index_tip, point_radius)

    if not caption:
        return

    text_surface = _render_text(caption, color, font_size)
    anchor_x = (thumb_tip[0] + index_tip[0]) // 2
    anchor_y = (thumb_tip[1] + index_tip[1]) // 2
    place_right = anchor_x < width * 0.5
    text_x = anchor_x + text_gap if place_right else anchor_x - text_surface.get_width() - text_gap
    text_y = anchor_y - text_surface.get_height() // 2
    text_x = int(_clamp(text_x, 8, width - text_surface.get_width() - 8))
    text_y = int(_clamp(text_y, 8, height - text_surface.get_height() - 8))
    surface.blit(text_surface, (text_x, text_y))
