from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import pygame
from PIL import Image, ImageDraw, ImageFont

from config import (
    HELP_LINES,
    HUD_BAR_BG,
    HUD_BAR_FILL,
    HUD_HELP_TEXT,
    HUD_MUTED,
    HUD_PANEL_BORDER,
    HUD_PANEL_COLOR,
    HUD_TEXT,
    LUMINOSITY_RANGE,
    MAX_TRAIL,
    MIN_TRAIL,
    PARTICLE_COUNT_RANGE,
    SCALE_RANGE,
    SPEED_RANGE,
    STATUS_LOST,
    STATUS_OK,
)

CONTROL_SLIDERS = (
    ("speed", "Speed", SPEED_RANGE[0], SPEED_RANGE[1], 0.05),
    ("trail_len", "Trail length", float(MIN_TRAIL), float(MAX_TRAIL), 100.0),
    ("particle_count", "Particles", float(PARTICLE_COUNT_RANGE[0]), float(PARTICLE_COUNT_RANGE[1]), 1.0),
    ("luminosity", "Luminosity", LUMINOSITY_RANGE[0], LUMINOSITY_RANGE[1], 0.01),
    ("scale", "Scale", SCALE_RANGE[0], SCALE_RANGE[1], 0.05),
)


class HUDRenderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.scale = max(0.62, min(width / 1920.0, height / 1080.0))
        self._navigation_count = 7
        self._font_cache: Dict[Tuple[int, str], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._surface_cache: Dict[Tuple[str, Tuple[int, int, int], int, str], pygame.Surface] = {}
        assets_fonts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts")
        home_fonts_dir = os.path.expanduser("~/Library/Fonts")
        self._font_candidates = {
            "sans": [
                os.path.join(assets_fonts_dir, "PlusJakartaSans-Regular.ttf"),
                os.path.join(assets_fonts_dir, "PlusJakartaSans[wght].ttf"),
                os.path.join(assets_fonts_dir, "PlusJakartaSans-VariableFont_wght.ttf"),
                os.path.join(home_fonts_dir, "PlusJakartaSans-Regular.ttf"),
                os.path.join(home_fonts_dir, "PlusJakartaSans[wght].ttf"),
                os.path.join(home_fonts_dir, "PlusJakartaSans-VariableFont_wght.ttf"),
                os.path.join("/Library/Fonts", "PlusJakartaSans-Regular.ttf"),
                os.path.join("/Library/Fonts", "PlusJakartaSans-VariableFont_wght.ttf"),
            ],
            "sans_bold": [
                os.path.join(assets_fonts_dir, "PlusJakartaSans-SemiBold.ttf"),
                os.path.join(assets_fonts_dir, "PlusJakartaSans-Bold.ttf"),
                os.path.join(assets_fonts_dir, "PlusJakartaSans[wght].ttf"),
                os.path.join(assets_fonts_dir, "PlusJakartaSans-VariableFont_wght.ttf"),
                os.path.join(home_fonts_dir, "PlusJakartaSans-SemiBold.ttf"),
                os.path.join(home_fonts_dir, "PlusJakartaSans-Bold.ttf"),
                os.path.join(home_fonts_dir, "PlusJakartaSans[wght].ttf"),
                os.path.join(home_fonts_dir, "PlusJakartaSans-VariableFont_wght.ttf"),
                os.path.join("/Library/Fonts", "PlusJakartaSans-SemiBold.ttf"),
                os.path.join("/Library/Fonts", "PlusJakartaSans-Bold.ttf"),
                os.path.join("/Library/Fonts", "PlusJakartaSans-VariableFont_wght.ttf"),
            ],
            "serif": [
                os.path.join(home_fonts_dir, "CormorantGaramond-Regular.ttf"),
                os.path.join("/Library/Fonts", "CormorantGaramond-Regular.ttf"),
                os.path.join("/System/Library/Fonts/Supplemental", "Baskerville.ttc"),
                os.path.join("/System/Library/Fonts/Supplemental", "Georgia.ttf"),
                os.path.join("/System/Library/Fonts/Supplemental", "Times New Roman.ttf"),
            ],
            "serif_italic": [
                os.path.join(home_fonts_dir, "CormorantGaramond-Italic.ttf"),
                os.path.join("/Library/Fonts", "CormorantGaramond-Italic.ttf"),
                os.path.join("/System/Library/Fonts/Supplemental", "Baskerville Italic.ttf"),
                os.path.join("/System/Library/Fonts/Supplemental", "Georgia Italic.ttf"),
                os.path.join("/System/Library/Fonts/Supplemental", "Times New Roman Italic.ttf"),
            ],
            "mono": [
                os.path.join("/System/Library/Fonts/Supplemental", "Menlo.ttc"),
                os.path.join("/Library/Fonts", "Courier New.ttf"),
                os.path.join("/System/Library/Fonts/Supplemental", "Courier New.ttf"),
            ],
        }

    def draw(
        self,
        surface: pygame.Surface,
        *,
        attractor_name: str,
        attractor_color: Tuple[int, int, int],
        placard_title: str,
        placard_year: str,
        placard_medium: str,
        placard_params: Sequence[Tuple[str, str]],
        attractor_names: Sequence[str],
        attractor_index: int,
        attractor_state: Tuple[float, float, float],
        point_count: int,
        speed: float,
        scale: float,
        luminosity: float,
        trail_len: int,
        particle_count: int,
        fps: float,
        left_detected: bool,
        right_detected: bool,
        show_overlay: bool,
        focus_mode: bool,
        active_slider: str | None,
        particle_input_active: bool,
        particle_input_text: str,
    ) -> None:
        self._navigation_count = len(attractor_names)
        if not focus_mode:
            self._draw_coordinates(surface, attractor_state)
            self._draw_navigation(surface, attractor_names, attractor_index, attractor_color)
            self._draw_placard(
                surface,
                attractor_color=attractor_color,
                attractor_index=attractor_index,
                placard_title=placard_title,
                placard_year=placard_year,
                placard_medium=placard_medium,
                placard_params=placard_params,
                point_count=point_count,
                particle_count=particle_count,
                particle_input_active=particle_input_active,
                particle_input_text=particle_input_text,
            )
            self._draw_control_panel(
                surface,
                attractor_color=attractor_color,
                speed=speed,
                trail_len=trail_len,
                particle_count=particle_count,
                luminosity=luminosity,
                scale=scale,
                active_slider=active_slider,
            )
        self._draw_footer(surface, fps)

        if show_overlay and not focus_mode:
            self._draw_operator_overlay(
                surface,
                speed=speed,
                scale=scale,
                luminosity=luminosity,
                left_detected=left_detected,
                right_detected=right_detected,
            )

    def control_hit_test(self, position: Tuple[int, int]) -> str | None:
        panel_rect, tracks = self._control_layout()
        if not panel_rect.collidepoint(position):
            return None
        for slider_id, rect in tracks.items():
            if rect.inflate(0, self._scaled(20)).collidepoint(position):
                return slider_id
        return None

    def control_value_for_position(self, slider_id: str, x_position: int) -> float:
        _, tracks = self._control_layout()
        track = tracks[slider_id]
        minimum, maximum, step = self._slider_spec(slider_id)
        ratio = self._clamp((x_position - track.left) / max(1, track.width), 0.0, 1.0)
        raw_value = minimum + (maximum - minimum) * ratio
        snapped = minimum + round((raw_value - minimum) / step) * step
        snapped = self._clamp(snapped, minimum, maximum)
        if slider_id in {"trail_len", "particle_count"}:
            return float(int(round(snapped)))
        return round(snapped, 2)

    def _draw_coordinates(self, surface: pygame.Surface, attractor_state: Tuple[float, float, float]) -> None:
        labels = ("x", "y", "z")
        x = self._scaled(72)
        y = self._scaled(60)
        gap = self._scaled(22)
        for idx, label in enumerate(labels):
            label_surface = self._render_text(label, HUD_MUTED, self._scaled(13), "mono")
            value_surface = self._render_text(f"{attractor_state[idx]:>7.3f}", HUD_TEXT, self._scaled(22), "mono")
            surface.blit(label_surface, (x, y))
            surface.blit(value_surface, (x, y + label_surface.get_height() + self._scaled(8)))
            x += max(label_surface.get_width(), value_surface.get_width()) + gap

    def _draw_navigation(
        self,
        surface: pygame.Surface,
        attractor_names: Sequence[str],
        active_index: int,
        attractor_color: Tuple[int, int, int],
    ) -> None:
        labels = self._navigation_labels()
        x, y, spacing, _ = self._navigation_layout()
        rendered_names = [
            self._render_text(label, attractor_color if idx == active_index else HUD_MUTED, self._scaled(20), "serif")
            for idx, label in enumerate(labels)
        ]
        for idx, rendered in enumerate(rendered_names):
            active = idx == active_index
            text_x = x
            text_y = y + idx * spacing
            surface.blit(rendered, (text_x, text_y))
            if active:
                underline_y = text_y + rendered.get_height() + self._scaled(4)
                pygame.draw.line(surface, attractor_color, (text_x, underline_y), (text_x + rendered.get_width() - 4, underline_y), 1)

    def _draw_placard(
        self,
        surface: pygame.Surface,
        *,
        attractor_color: Tuple[int, int, int],
        attractor_index: int,
        placard_title: str,
        placard_year: str,
        placard_medium: str,
        placard_params: Sequence[Tuple[str, str]],
        point_count: int,
        particle_count: int,
        particle_input_active: bool,
        particle_input_text: str,
    ) -> None:
        x = self._scaled(72)
        placard_width = min(self._scaled(420), int(self.width * 0.26))
        study_line = self._render_text(f"Study no. {self._roman(attractor_index + 1)}", HUD_MUTED, self._scaled(15), "mono")
        title_lines = self._wrap_lines(placard_title, self._scaled(34), placard_width, "serif_italic", attractor_color)
        year_lines = self._wrap_lines(placard_year, self._scaled(18), placard_width, "serif", HUD_MUTED)
        medium_lines = self._wrap_lines(placard_medium, self._scaled(14), placard_width, "serif_italic")

        display_rows = list(placard_params[:3])
        display_rows.append(("particles", str(particle_count)))
        points_value = f"[{particle_input_text}_]" if particle_input_active else f"{point_count:,}"
        display_rows.append(("points", points_value))
        row_height = self._scaled(22)

        content_height = self._scaled(18)
        content_height += study_line.get_height()
        content_height += self._scaled(12)
        content_height += self._lines_height(title_lines, self._scaled(4))
        content_height += self._scaled(6)
        content_height += self._lines_height(year_lines, self._scaled(4))
        content_height += self._scaled(14)
        content_height += self._lines_height(medium_lines, self._scaled(4))
        content_height += self._scaled(16)
        content_height += 1
        content_height += self._scaled(16)
        content_height += row_height * len(display_rows)

        y = max(self._scaled(170), self.height - self._scaled(84) - content_height)

        pygame.draw.line(surface, HUD_BAR_FILL, (x, y), (x + self._scaled(24), y), 1)
        cursor_y = y + self._scaled(18)
        surface.blit(study_line, (x, cursor_y))
        cursor_y += study_line.get_height() + self._scaled(12)
        cursor_y = self._blit_lines(surface, title_lines, x, cursor_y, self._scaled(4))
        cursor_y += self._scaled(6)
        cursor_y = self._blit_lines(surface, year_lines, x, cursor_y, self._scaled(4))
        cursor_y += self._scaled(14)
        cursor_y = self._blit_lines(surface, medium_lines, x, cursor_y, self._scaled(4))
        cursor_y += self._scaled(16)

        pygame.draw.line(surface, HUD_PANEL_BORDER, (x, cursor_y), (x + placard_width, cursor_y), 1)
        cursor_y += self._scaled(16)

        for label, value in display_rows:
            label_surface = self._render_text(label, HUD_MUTED, self._scaled(14), "mono")
            value_surface = self._render_text(value, HUD_TEXT, self._scaled(15), "mono")
            surface.blit(label_surface, (x, cursor_y))
            surface.blit(value_surface, (x + placard_width - value_surface.get_width(), cursor_y - self._scaled(1)))
            cursor_y += row_height

    def _draw_control_panel(
        self,
        surface: pygame.Surface,
        *,
        attractor_color: Tuple[int, int, int],
        speed: float,
        trail_len: int,
        particle_count: int,
        luminosity: float,
        scale: float,
        active_slider: str | None,
    ) -> None:
        panel_rect, tracks = self._control_layout()
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, HUD_PANEL_COLOR, panel.get_rect(), border_radius=self._scaled(18))
        pygame.draw.rect(panel, HUD_PANEL_BORDER, panel.get_rect(), 1, border_radius=self._scaled(18))
        surface.blit(panel, panel_rect.topleft)

        title = self._render_text("Parameters", HUD_TEXT, self._scaled(16), "mono")
        surface.blit(title, (panel_rect.left + self._scaled(24), panel_rect.top + self._scaled(18)))

        values = {
            "speed": speed,
            "trail_len": float(trail_len),
            "particle_count": float(particle_count),
            "luminosity": luminosity,
            "scale": scale,
        }
        for slider_id, label, *_ in CONTROL_SLIDERS:
            track = tracks[slider_id]
            row_top = track.top - self._scaled(24)
            label_surface = self._render_text(label, HUD_TEXT, self._scaled(15), "serif_italic")
            value_surface = self._render_text(self._format_slider_value(slider_id, values[slider_id]), HUD_BAR_FILL, self._scaled(13), "mono")
            surface.blit(label_surface, (track.left, row_top))
            surface.blit(value_surface, (track.right - value_surface.get_width(), row_top))

            active = active_slider == slider_id
            ratio = self._slider_ratio(slider_id, values[slider_id])
            fill_width = int(round(track.width * ratio))
            fill_width = max(0, min(track.width, fill_width))
            filled_rect = pygame.Rect(track.left, track.top, fill_width, track.height)

            pygame.draw.line(surface, HUD_BAR_BG, track.midleft, track.midright, max(1, track.height))
            if fill_width > 0:
                pygame.draw.line(surface, attractor_color, filled_rect.midleft, filled_rect.midright, max(1, track.height))

            knob_x = track.left + fill_width
            knob_x = max(track.left, min(track.right, knob_x))
            pygame.draw.circle(surface, attractor_color if active else HUD_BAR_FILL, (knob_x, track.centery), self._scaled(6 if active else 5))

    def _draw_footer(self, surface: pygame.Surface, fps: float) -> None:
        text = f"{fps:>5.1f} FPS"
        footer_size = self._scaled(13)
        footer = self._render_text(self._fit_text(text, footer_size, self.width - self._scaled(96), "mono"), HUD_MUTED, footer_size, "mono")
        footer_y = self.height - self._scaled(34)
        surface.blit(footer, ((self.width - footer.get_width()) // 2, footer_y))

    def _draw_operator_overlay(
        self,
        surface: pygame.Surface,
        *,
        speed: float,
        scale: float,
        luminosity: float,
        left_detected: bool,
        right_detected: bool,
    ) -> None:
        content_width = min(self._scaled(350), int(self.width * 0.24))
        wrapped_blocks: list[tuple[Tuple[int, int, int], str, list[str]] | None] = []
        for line in HELP_LINES:
            if not line:
                wrapped_blocks.append(None)
                continue
            color = HUD_HELP_TEXT if line.isupper() else HUD_MUTED
            style = "sans_bold" if line.isupper() else "sans"
            wrapped_blocks.append((color, style, self._wrap_text(line, self._scaled(14), content_width, style)))

        width = content_width + self._scaled(32)
        line_gap = self._scaled(4)
        section_gap = self._scaled(10)
        content_height = 0
        for block in wrapped_blocks:
            if block is None:
                content_height += section_gap
                continue
            _, style, lines = block
            line_height = self._font_line_height(self._scaled(14), style)
            content_height += line_height * len(lines) + line_gap * max(0, len(lines) - 1) + self._scaled(6)

        stats_bottom = self._scaled(150)
        height = self._scaled(24) + self._scaled(18) + self._scaled(24) + stats_bottom + content_height + self._scaled(16)
        x = self._scaled(28)
        y = self._scaled(120)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(HUD_PANEL_COLOR)
        pygame.draw.rect(panel, HUD_PANEL_BORDER, panel.get_rect(), 1)
        surface.blit(panel, (x, y))

        title = self._render_text("Operator Overlay", HUD_TEXT, self._scaled(18), "sans_bold")
        surface.blit(title, (x + self._scaled(16), y + self._scaled(14)))
        # The camera preview is mirrored, so the UI labels are presented
        # from the viewer's perspective rather than MediaPipe slot names.
        self._draw_status(surface, "RIGHT", left_detected, x + self._scaled(18), y + self._scaled(48))
        self._draw_status(surface, "LEFT", right_detected, x + self._scaled(116), y + self._scaled(48))
        self._draw_metric(surface, "SPEED", speed, x + self._scaled(18), y + self._scaled(84))
        self._draw_metric(surface, "SCALE", scale, x + self._scaled(18), y + self._scaled(106))
        self._draw_metric(surface, "LUM", luminosity, x + self._scaled(18), y + self._scaled(128))

        cursor_y = y + self._scaled(166)
        for block in wrapped_blocks:
            if block is None:
                cursor_y += section_gap
                continue
            color, style, lines = block
            for wrapped_line in lines:
                rendered = self._render_text(wrapped_line, color, self._scaled(14), style)
                surface.blit(rendered, (x + self._scaled(16), cursor_y))
                cursor_y += rendered.get_height() + line_gap
            cursor_y += self._scaled(2)

    def _draw_status(self, surface: pygame.Surface, label: str, active: bool, x: int, y: int) -> None:
        color = STATUS_OK if active else STATUS_LOST
        pygame.draw.circle(surface, color, (x + self._scaled(6), y + self._scaled(8)), self._scaled(5))
        text = self._render_text(label, HUD_TEXT, self._scaled(13), "mono")
        surface.blit(text, (x + self._scaled(20), y))

    def _draw_metric(self, surface: pygame.Surface, label: str, value: float, x: int, y: int) -> None:
        label_surface = self._render_text(label, HUD_MUTED, self._scaled(13), "mono")
        value_surface = self._render_text(f"{value:>4.2f}", HUD_TEXT, self._scaled(13), "mono")
        surface.blit(label_surface, (x, y))
        surface.blit(value_surface, (x + self._scaled(204), y))

    def _control_layout(self) -> tuple[pygame.Rect, Dict[str, pygame.Rect]]:
        nav_x, nav_y, _spacing, nav_width = self._navigation_layout()
        panel_width = min(self._scaled(320), max(self._scaled(250), int(self.width * 0.19)))
        panel_height = self._scaled(100 + 44 * (len(CONTROL_SLIDERS) - 1))
        panel_x = max(self._scaled(72), nav_x - self._scaled(28) - panel_width)
        panel_y = nav_y
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        track_left = panel_rect.left + self._scaled(24)
        track_width = panel_rect.width - self._scaled(48)
        row_start_y = panel_rect.top + self._scaled(52)
        row_gap = self._scaled(44)
        track_height = max(2, self._scaled(2))
        tracks: Dict[str, pygame.Rect] = {}
        for index, (slider_id, *_rest) in enumerate(CONTROL_SLIDERS):
            track_y = row_start_y + index * row_gap
            tracks[slider_id] = pygame.Rect(track_left, track_y, track_width, track_height)
        return panel_rect, tracks

    def _navigation_labels(self) -> tuple[str, ...]:
        return tuple(f"figure. {idx + 1}" for idx in range(self._navigation_count))

    def _navigation_layout(self) -> tuple[int, int, int, int]:
        labels = self._navigation_labels()
        rendered = [self._render_text(label, HUD_MUTED, self._scaled(20), "serif") for label in labels]
        max_width = max(surface.get_width() for surface in rendered)
        spacing = max(surface.get_height() for surface in rendered) + self._scaled(14)
        x = self.width - self._scaled(72) - max_width
        y = self._scaled(96)
        return x, y, spacing, max_width

    def _slider_spec(self, slider_id: str) -> tuple[float, float, float]:
        for current_id, _label, minimum, maximum, step in CONTROL_SLIDERS:
            if current_id == slider_id:
                return minimum, maximum, step
        raise KeyError(slider_id)

    def _slider_ratio(self, slider_id: str, value: float) -> float:
        minimum, maximum, _step = self._slider_spec(slider_id)
        return self._clamp((value - minimum) / max(0.0001, maximum - minimum), 0.0, 1.0)

    def _format_slider_value(self, slider_id: str, value: float) -> str:
        if slider_id in {"trail_len", "particle_count"}:
            return str(int(round(value)))
        return f"{value:0.2f}"

    def _scaled(self, value: int) -> int:
        return max(1, int(round(value * self.scale)))

    def _wrap_text(self, text: str, size: int, max_width: int, style: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self._text_width(candidate, size, style) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _wrap_lines(
        self,
        text: str,
        size: int,
        max_width: int,
        style: str,
        color: Tuple[int, int, int] | None = None,
    ) -> list[pygame.Surface]:
        wrapped: list[str] = []
        for paragraph in text.splitlines() or [text]:
            wrapped.extend(self._wrap_text(paragraph, size, max_width, style))
        render_color = color or (HUD_TEXT if style == "serif" else HUD_MUTED)
        return [self._render_text(line, render_color, size, style) for line in wrapped]

    def _lines_height(self, lines: Sequence[pygame.Surface], gap: int) -> int:
        if not lines:
            return 0
        return sum(line.get_height() for line in lines) + gap * (len(lines) - 1)

    def _blit_lines(self, surface: pygame.Surface, lines: Sequence[pygame.Surface], x: int, y: int, gap: int) -> int:
        cursor_y = y
        for idx, line in enumerate(lines):
            surface.blit(line, (x, cursor_y))
            cursor_y += line.get_height()
            if idx < len(lines) - 1:
                cursor_y += gap
        return cursor_y

    def _roman(self, value: int) -> str:
        numerals = (
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        )
        result = []
        remaining = value
        for arabic, roman in numerals:
            while remaining >= arabic:
                result.append(roman)
                remaining -= arabic
        return "".join(result)

    def _font(self, size: int, style: str):
        key = (size, style)
        if key not in self._font_cache:
            font = None
            for path in self._font_candidates.get(style, self._font_candidates["sans"]):
                if not os.path.exists(path):
                    continue
                try:
                    font = ImageFont.truetype(path, size=size)
                    if "VariableFont" in os.path.basename(path) or "[wght]" in os.path.basename(path):
                        variation_name = "SemiBold" if style == "sans_bold" else "Regular"
                        font.set_variation_by_name(variation_name)
                    break
                except OSError:
                    continue
            self._font_cache[key] = font or ImageFont.load_default(size=size)
        return self._font_cache[key]

    def _font_line_height(self, size: int, style: str) -> int:
        font = self._font(size, style)
        bbox = font.getbbox("Ag")
        return max(1, bbox[3] - bbox[1] + 2)

    def _render_text(self, text: str, color: Tuple[int, int, int], size: int, style: str) -> pygame.Surface:
        key = (text, color, size, style)
        cached = self._surface_cache.get(key)
        if cached is not None:
            return cached

        font = self._font(size, style)
        bbox = font.getbbox(text or " ")
        width = max(1, bbox[2] - bbox[0] + 2)
        height = max(1, bbox[3] - bbox[1] + 2)

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text((1 - bbox[0], 1 - bbox[1]), text, font=font, fill=(*color, 255))
        rendered = pygame.image.fromstring(image.tobytes(), image.size, image.mode).convert_alpha()
        self._surface_cache[key] = rendered
        return rendered

    def _text_width(self, text: str, size: int, style: str) -> int:
        font = self._font(size, style)
        bbox = font.getbbox(text or " ")
        return max(1, bbox[2] - bbox[0] + 2)

    def _fit_text(self, text: str, size: int, max_width: int, style: str) -> str:
        if self._text_width(text, size, style) <= max_width:
            return text

        ellipsis = "..."
        trimmed = text
        while trimmed and self._text_width(trimmed + ellipsis, size, style) > max_width:
            trimmed = trimmed[:-1]
        return (trimmed + ellipsis) if trimmed else ellipsis

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))
