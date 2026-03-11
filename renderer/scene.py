from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pygame

from config import (
    BACKGROUND_COLOR,
    CAPTION,
    DEFAULT_POINT_SIZE,
    FPS,
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
    PIP_H,
    PIP_MARGIN,
    PIP_W,
    SCALE_RANGE,
    SPEED_RANGE,
    WIN_H,
    WIN_W,
)
from hands.skeleton import draw_hand_skeleton

from .common import compute_mvp


VERTEX_SHADER = """
#version 330

uniform mat4 u_mvp;
uniform float u_time;
uniform float u_point_scale;

in vec3 in_position;
in float in_age;

out float v_age;

mat3 rotationY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c
    );
}

void main() {
    float pulse = 1.0 + 0.045 * sin(u_time * 0.85 + in_age * 4.0);
    float drift = sin(u_time * 0.33) * 0.32;
    vec3 animated = rotationY(drift) * (in_position * pulse);
    vec4 clip = u_mvp * vec4(animated, 1.0);
    gl_Position = clip;

    float depth = max(0.30, clip.w);
    gl_PointSize = clamp((u_point_scale / depth) + 0.8 + in_age * 2.0, 1.0, 7.5);
    v_age = in_age;
}
"""


FRAGMENT_SHADER = """
#version 330

uniform vec3 u_color;
uniform float u_luminosity;

in float v_age;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) {
        discard;
    }

    float halo = exp(-4.2 * r2);
    float core = pow(max(0.0, 1.0 - r2), 7.0);
    float glow = max(core, halo * 0.75);
    float alpha = glow * mix(0.018, 0.15, v_age) * max(u_luminosity, 0.05);
    vec3 color = mix(u_color * 0.55, vec3(1.0), v_age * 0.28);
    fragColor = vec4(color, alpha);
}
"""


QUAD_VERTEX_SHADER = """
#version 330

uniform vec2 u_screen_size;
uniform vec4 u_rect;

in vec2 in_pos;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    vec2 pixel = u_rect.xy + in_pos * u_rect.zw;
    vec2 ndc = vec2(
        (pixel.x / u_screen_size.x) * 2.0 - 1.0,
        1.0 - (pixel.y / u_screen_size.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""


QUAD_FRAGMENT_SHADER = """
#version 330

uniform sampler2D u_texture;
uniform float u_opacity;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec4 color = texture(u_texture, v_uv);
    fragColor = vec4(color.rgb, color.a * u_opacity);
}
"""

CONTROL_SLIDERS = (
    ("speed", SPEED_RANGE),
    ("trail_len", (float(MIN_TRAIL), float(MAX_TRAIL))),
    ("luminosity", LUMINOSITY_RANGE),
    ("scale", SCALE_RANGE),
)
RESET_TRAIL_ACTION = "reset_trail"


@dataclass
class SceneState:
    positions: np.ndarray
    ages: np.ndarray
    attractor_name: str
    attractor_color: tuple[int, int, int]
    attractor_index: int
    attractor_total: int
    attractor_names: Sequence[str]
    attractor_state: tuple[float, float, float]
    placard_title: str
    placard_year: str
    placard_medium: str
    placard_params: Sequence[tuple[str, str]]
    yaw: float
    pitch: float
    roll: float
    zoom: float
    speed: float
    luminosity: float
    trail_len: int
    point_count: int
    fps: float
    paused: bool
    exporting: bool
    export_message: str
    show_overlay: bool
    show_camera: bool
    focus_mode: bool
    active_slider: Optional[str]
    left_detected: bool
    right_detected: bool
    pip_frame: Optional[np.ndarray]
    left_landmarks: Optional[Sequence[tuple[float, float, float]]]
    right_landmarks: Optional[Sequence[tuple[float, float, float]]]
    left_pip_caption: str
    right_pip_caption: str
    time_value: float


class SceneRenderer:
    def __init__(self, width: int = WIN_W, height: int = WIN_H) -> None:
        import moderngl

        self.moderngl = moderngl
        self.width = width
        self.height = height
        self._point_capacity = 16
        self._overlay_size = (width, height)
        self._cv2 = None

        pygame.init()
        pygame.display.set_caption(CAPTION)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        try:
            pygame.display.set_mode((width, height), flags, vsync=1)
        except TypeError:  # pragma: no cover - pygame < 2.1 fallback
            pygame.display.set_mode((width, height), flags)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        if hasattr(moderngl, "PROGRAM_POINT_SIZE"):
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        self.point_program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.point_buffer = self.ctx.buffer(reserve=self._point_capacity)
        self.point_vao = self.ctx.vertex_array(self.point_program, [(self.point_buffer, "3f 1f", "in_position", "in_age")])

        self.quad_program = self.ctx.program(vertex_shader=QUAD_VERTEX_SHADER, fragment_shader=QUAD_FRAGMENT_SHADER)
        quad_vertices = np.array(
            [
                0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        self.quad_buffer = self.ctx.buffer(quad_vertices.tobytes())
        self.quad_vao = self.ctx.vertex_array(self.quad_program, [(self.quad_buffer, "2f 2f", "in_pos", "in_uv")])

        self.overlay_texture = self.ctx.texture(self._overlay_size, 4)
        self.overlay_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.camera_texture = self.ctx.texture((PIP_W, PIP_H), 3)
        self.camera_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._font_mono_small = self._load_font(12, "mono")
        self._font_mono = self._load_font(16, "mono")
        self._font_body = self._load_font(16, "sans")
        self._font_body_bold = self._load_font(17, "sans_bold")
        self._font_serif = self._load_font(18, "serif")
        self._font_title = self._load_font(30, "serif_italic")
        self.clock = pygame.time.Clock()

    def _load_font(self, size: int, style: str):
        families = {
            "mono": [
                "/System/Library/Fonts/Supplemental/Menlo.ttc",
                "/Library/Fonts/Courier New.ttf",
                "DejaVuSansMono.ttf",
            ],
            "sans": [
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                "DejaVuSans.ttf",
            ],
            "sans_bold": [
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "DejaVuSans-Bold.ttf",
            ],
            "serif": [
                "/System/Library/Fonts/Supplemental/Georgia.ttf",
                "/System/Library/Fonts/Supplemental/Baskerville.ttc",
                "DejaVuSerif.ttf",
            ],
            "serif_italic": [
                "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
                "/System/Library/Fonts/Supplemental/Baskerville Italic.ttf",
                "DejaVuSerif-Italic.ttf",
            ],
        }
        for candidate in families.get(style, families["sans"]):
            if os.path.isabs(candidate) and not os.path.exists(candidate):
                continue
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _sync_window_size(self) -> None:
        width, height = pygame.display.get_window_size()
        if (width, height) != (self.width, self.height):
            self.width, self.height = width, height
        self.ctx.viewport = (0, 0, self.width, self.height)
        if (self.width, self.height) != self._overlay_size:
            self.overlay_texture.release()
            self._overlay_size = (self.width, self.height)
            self.overlay_texture = self.ctx.texture(self._overlay_size, 4)
            self.overlay_texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)

    def _ensure_point_capacity(self, byte_count: int) -> None:
        if byte_count <= self._point_capacity:
            return
        self._point_capacity = max(byte_count, self._point_capacity * 2)
        self.point_buffer.release()
        self.point_buffer = self.ctx.buffer(reserve=self._point_capacity)
        self.point_vao.release()
        self.point_vao = self.ctx.vertex_array(self.point_program, [(self.point_buffer, "3f 1f", "in_position", "in_age")])

    def _render_quad_texture(self, texture, rect: tuple[int, int, int, int], opacity: float = 1.0) -> None:
        self.ctx.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE_MINUS_SRC_ALPHA
        texture.use(location=0)
        self.quad_program["u_texture"].value = 0
        self.quad_program["u_opacity"].value = float(opacity)
        self.quad_program["u_screen_size"].value = (float(self.width), float(self.height))
        self.quad_program["u_rect"].value = tuple(float(value) for value in rect)
        self.quad_vao.render()

    def draw(self, state: SceneState) -> None:
        self._sync_window_size()
        background = tuple(channel / 255.0 for channel in BACKGROUND_COLOR)
        self.ctx.clear(background[0], background[1], background[2], 1.0)

        if len(state.positions) > 0:
            vertex_data = np.empty((len(state.positions), 4), dtype=np.float32)
            vertex_data[:, :3] = state.positions
            vertex_data[:, 3] = state.ages
            self._ensure_point_capacity(vertex_data.nbytes)
            self.point_buffer.write(vertex_data.tobytes())
            mvp = compute_mvp(
                self.width,
                self.height,
                yaw=state.yaw,
                pitch=state.pitch,
                roll=state.roll,
                zoom=state.zoom,
            )
            self.point_program["u_mvp"].write(mvp.T.astype(np.float32, copy=False).tobytes())
            self.point_program["u_time"].value = float(state.time_value)
            self.point_program["u_point_scale"].value = float(DEFAULT_POINT_SIZE)
            self.point_program["u_luminosity"].value = float(state.luminosity)
            self.point_program["u_color"].value = tuple(channel / 255.0 for channel in state.attractor_color)
            self.ctx.blend_func = self.moderngl.ONE, self.moderngl.ONE
            self.point_vao.render(mode=self.moderngl.POINTS, vertices=len(state.positions))

        if state.show_overlay and not state.focus_mode:
            self._draw_overlay(state)
        if (state.show_camera or state.focus_mode) and state.pip_frame is not None:
            self._draw_camera_frame(
                state.pip_frame,
                state.left_landmarks,
                state.right_landmarks,
                state.left_pip_caption,
                state.right_pip_caption,
            )

    def _draw_overlay(self, state: SceneState) -> None:
        image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        accent = tuple(state.attractor_color)
        self._draw_coordinates(draw, state)
        self._draw_helper_panel(draw, state, accent)
        self._draw_navigation(draw, state, accent)
        self._draw_control_panel(draw, state, accent)
        self._draw_placard(draw, state, accent)
        self._draw_footer(draw, state)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.overlay_texture.write(image.tobytes())
        self._render_quad_texture(self.overlay_texture, (0, 0, self.width, self.height))

    def _draw_coordinates(self, draw: ImageDraw.ImageDraw, state: SceneState) -> None:
        labels = ("x", "y", "z")
        x = 48
        y = 34
        gap = 74
        for idx, label in enumerate(labels):
            draw.text((x, y), label, font=self._font_mono_small, fill=HUD_MUTED)
            value = f"{state.attractor_state[idx]:>7.3f}"
            draw.text((x + 18, y + 12), value, font=self._font_mono, fill=HUD_TEXT)
            x += gap

    def _draw_navigation(self, draw: ImageDraw.ImageDraw, state: SceneState, accent: tuple[int, int, int]) -> None:
        x, y, spacing, _ = self._navigation_metrics(state)
        for idx, _name in enumerate(state.attractor_names):
            label = f"figure. {idx + 1}"
            color = accent if idx == state.attractor_index else HUD_MUTED
            draw.text((x, y + idx * spacing), label, font=self._font_serif, fill=color)

    def _navigation_metrics(self, state: SceneState) -> tuple[int, int, int, int]:
        labels = [f"figure. {idx + 1}" for idx in range(len(state.attractor_names))]
        max_width = max(self._text_width(label, self._font_serif) for label in labels)
        x = self.width - 54 - max_width
        y = 58
        spacing = 33
        return x, y, spacing, max_width

    def _draw_control_panel(self, draw: ImageDraw.ImageDraw, state: SceneState, accent: tuple[int, int, int]) -> None:
        panel_rect, track_rects, button_rect = self._control_panel_layout(state.attractor_names)
        panel_x, panel_y, panel_right, panel_bottom = panel_rect
        panel_width = panel_right - panel_x
        panel_height = panel_bottom - panel_y
        fill = (*HUD_PANEL_COLOR[:3], 220)
        border = (*HUD_PANEL_BORDER, 255)
        draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_width, panel_y + panel_height), radius=18, fill=fill, outline=border, width=1)
        draw.text((panel_x + 18, panel_y + 14), "Parameters", font=self._font_mono_small, fill=HUD_TEXT)

        rows = {
            "speed": ("Speed", state.speed, SPEED_RANGE),
            "trail_len": ("Trail length", float(state.trail_len), (float(MIN_TRAIL), float(MAX_TRAIL))),
            "luminosity": ("Luminosity", state.luminosity, LUMINOSITY_RANGE),
            "scale": ("Scale", state.zoom, SCALE_RANGE),
        }
        for slider_id, (_value_range) in CONTROL_SLIDERS:
            label, value, value_range = rows[slider_id]
            track_left, track_top, track_right, _track_bottom = track_rects[slider_id]
            self._draw_slider_row(
                draw,
                track_left,
                track_top - 20,
                track_right - track_left,
                label,
                float(value),
                value_range,
                accent,
                active=state.active_slider == slider_id,
            )

        self._draw_action_button(draw, button_rect, "Reset trail", accent)
        status = "paused" if state.paused else "exporting..." if state.exporting else state.export_message or "live"
        status_fill = accent if not state.paused else HUD_BAR_FILL
        draw.text((panel_x + 18, panel_y + panel_height - 24), status, font=self._font_mono_small, fill=status_fill)

    def _draw_helper_panel(self, draw: ImageDraw.ImageDraw, state: SceneState, accent: tuple[int, int, int]) -> None:
        panel_x = 44
        panel_y = 104
        panel_width = 248
        help_lines = [
            "Overlay helper",
            "1-7 switch attractor",
            "R restart trail growth",
            "SPACE pause or resume",
            "S save 4K snapshot",
            "Mouse wheel zoom",
            "Left hand: yaw pitch speed",
            "Right hand: glow zoom",
            "Pinky touch: switch scene",
        ]
        line_height = 18
        panel_height = 28 + len(help_lines) * line_height + 14
        fill = (*HUD_PANEL_COLOR[:3], 205)
        border = (*HUD_PANEL_BORDER, 255)
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
            radius=16,
            fill=fill,
            outline=border,
            width=1,
        )
        cursor_y = panel_y + 14
        for idx, line in enumerate(help_lines):
            if idx == 0:
                draw.text((panel_x + 16, cursor_y), line, font=self._font_body_bold, fill=accent)
            else:
                draw.text((panel_x + 16, cursor_y), line, font=self._font_mono_small, fill=HUD_HELP_TEXT)
            cursor_y += line_height

    def _draw_slider_row(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        value_range: tuple[float, float],
        accent: tuple[int, int, int],
        *,
        active: bool,
    ) -> None:
        minimum, maximum = value_range
        ratio = 0.0 if maximum <= minimum else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
        label_y = y
        track_y = y + 20
        value_text = f"{value:0.2f}" if label != "Trail length" else f"{int(round(value))}"
        draw.text((x, label_y), label, font=self._font_serif, fill=HUD_TEXT)
        value_w = self._text_width(value_text, self._font_mono_small)
        draw.text((x + width - value_w, label_y + 1), value_text, font=self._font_mono_small, fill=HUD_BAR_FILL)
        draw.line((x, track_y, x + width, track_y), fill=HUD_BAR_BG, width=2)
        fill_end = x + int(width * ratio)
        if fill_end > x:
            draw.line((x, track_y, fill_end, track_y), fill=accent, width=2)
        knob_radius = 5 if active else 4
        knob_fill = accent if active else HUD_BAR_FILL
        draw.ellipse((fill_end - knob_radius, track_y - knob_radius, fill_end + knob_radius, track_y + knob_radius), fill=knob_fill)

    def _draw_action_button(
        self,
        draw: ImageDraw.ImageDraw,
        rect: tuple[int, int, int, int],
        label: str,
        accent: tuple[int, int, int],
    ) -> None:
        left, top, right, bottom = rect
        fill = (*HUD_BAR_BG, 255)
        border = (*accent, 255)
        draw.rounded_rectangle((left, top, right, bottom), radius=12, fill=fill, outline=border, width=1)
        label_w = self._text_width(label, self._font_mono_small)
        label_x = left + max(0, (right - left - label_w) // 2)
        label_y = top + max(0, (bottom - top - 14) // 2)
        draw.text((label_x, label_y), label, font=self._font_mono_small, fill=HUD_TEXT)

    def _control_panel_layout(
        self,
        attractor_names: Sequence[str],
    ) -> tuple[tuple[int, int, int, int], dict[str, tuple[int, int, int, int]], tuple[int, int, int, int]]:
        labels = [f"figure. {idx + 1}" for idx in range(len(attractor_names))]
        max_width = max(self._text_width(label, self._font_serif) for label in labels) if labels else 0
        nav_x = self.width - 54 - max_width
        panel_width = 310
        panel_height = 252
        panel_gap = 34
        panel_x = max(48, nav_x - panel_gap - panel_width)
        panel_y = 60
        track_rects: dict[str, tuple[int, int, int, int]] = {}
        row_y = panel_y + 40
        for slider_id, _value_range in CONTROL_SLIDERS:
            track_left = panel_x + 18
            track_top = row_y + 20
            track_right = panel_x + panel_width - 18
            track_bottom = track_top + 10
            track_rects[slider_id] = (track_left, track_top, track_right, track_bottom)
            row_y += 38
        button_rect = (panel_x + 18, panel_y + 192, panel_x + panel_width - 18, panel_y + 224)
        return (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height), track_rects, button_rect

    def control_hit_test(self, attractor_names: Sequence[str], position: tuple[int, int]) -> Optional[str]:
        panel_rect, track_rects, _button_rect = self._control_panel_layout(attractor_names)
        x, y = position
        if not (panel_rect[0] <= x <= panel_rect[2] and panel_rect[1] <= y <= panel_rect[3]):
            return None
        for slider_id, (left, top, right, bottom) in track_rects.items():
            if left <= x <= right and top - 16 <= y <= bottom + 8:
                return slider_id
        return None

    def reset_button_hit_test(self, attractor_names: Sequence[str], position: tuple[int, int]) -> bool:
        _panel_rect, _track_rects, button_rect = self._control_panel_layout(attractor_names)
        x, y = position
        return button_rect[0] <= x <= button_rect[2] and button_rect[1] <= y <= button_rect[3]

    def control_value_for_position(self, attractor_names: Sequence[str], slider_id: str, x_position: int) -> float:
        _panel_rect, track_rects, _button_rect = self._control_panel_layout(attractor_names)
        left, _top, right, _bottom = track_rects[slider_id]
        minimum, maximum = dict(CONTROL_SLIDERS)[slider_id]
        ratio = 0.0 if right <= left else max(0.0, min(1.0, (x_position - left) / (right - left)))
        value = minimum + (maximum - minimum) * ratio
        if slider_id == "trail_len":
            return float(int(round(value / 100.0) * 100))
        return round(value, 2)

    def _draw_placard(self, draw: ImageDraw.ImageDraw, state: SceneState, accent: tuple[int, int, int]) -> None:
        x = 48
        placard_width = 275
        y = self.height - 238
        draw.line((x, y, x + 22, y), fill=HUD_BAR_FILL, width=1)
        cursor_y = y + 18
        draw.text((x, cursor_y), f"Study no. {self._roman(state.attractor_index + 1)}", font=self._font_mono_small, fill=HUD_MUTED)
        cursor_y += 26
        draw.text((x, cursor_y), state.placard_title, font=self._font_title, fill=accent)
        cursor_y += 34
        draw.text((x, cursor_y), state.placard_year, font=self._font_serif, fill=HUD_MUTED)
        cursor_y += 22
        medium_lines = self._wrap_lines(state.placard_medium, 34)
        for line in medium_lines:
            draw.text((x, cursor_y), line, font=self._font_serif, fill=HUD_HELP_TEXT)
            cursor_y += 16
        cursor_y += 10
        draw.line((x, cursor_y, x + placard_width, cursor_y), fill=HUD_PANEL_BORDER, width=1)
        cursor_y += 12
        rows = list(state.placard_params[:3]) + [("points", f"{state.point_count:,}")]
        for label, value in rows:
            draw.text((x, cursor_y), label, font=self._font_mono_small, fill=HUD_MUTED)
            value_w = self._text_width(value, self._font_mono_small)
            draw.text((x + placard_width - value_w, cursor_y), value, font=self._font_mono_small, fill=HUD_TEXT)
            cursor_y += 15

    def _draw_footer(self, draw: ImageDraw.ImageDraw, state: SceneState) -> None:
        footer = f"{state.fps:>4.1f} FPS"
        footer_w = self._text_width(footer, self._font_mono_small)
        draw.text(((self.width - footer_w) // 2, self.height - 28), footer, font=self._font_mono_small, fill=HUD_MUTED)

    def _wrap_lines(self, text: str, width: int) -> list[str]:
        lines: list[str] = []
        for paragraph in text.splitlines() or [text]:
            wrapped = textwrap.wrap(paragraph, width=width) or [paragraph]
            lines.extend(wrapped)
        return lines

    def _roman(self, value: int) -> str:
        numerals = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
        remaining = value
        output: list[str] = []
        for arabic, roman in numerals:
            while remaining >= arabic:
                output.append(roman)
                remaining -= arabic
        return "".join(output)

    def _text_width(self, text: str, font) -> int:
        bbox = font.getbbox(text or " ")
        return max(1, bbox[2] - bbox[0] + 2)

    def _draw_camera_frame(
        self,
        frame_bgr: np.ndarray,
        left_landmarks: Optional[Sequence[tuple[float, float, float]]],
        right_landmarks: Optional[Sequence[tuple[float, float, float]]],
        left_caption: str,
        right_caption: str,
    ) -> None:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError:  # pragma: no cover - optional dependency
                cv2 = None
            self._cv2 = cv2
        if self._cv2 is not None:
            preview = self._cv2.resize(frame_bgr, (PIP_W, PIP_H))
            preview_rgb = self._cv2.cvtColor(preview, self._cv2.COLOR_BGR2RGB)
        else:  # pragma: no cover - only used when cv2 is unavailable
            preview = np.asarray(frame_bgr)
            preview_rgb = np.resize(preview[:, :, ::-1], (PIP_H, PIP_W, 3))
        if left_landmarks or right_landmarks:
            overlay = pygame.Surface((PIP_W, PIP_H), pygame.SRCALPHA)
            if left_landmarks:
                draw_hand_skeleton(overlay, left_landmarks, PIP_W, PIP_H, caption=left_caption)
            if right_landmarks:
                draw_hand_skeleton(overlay, right_landmarks, PIP_W, PIP_H, caption=right_caption)
            overlay_rgba = pygame.image.tobytes(overlay, "RGBA")
            overlay_rgba = np.frombuffer(overlay_rgba, dtype=np.uint8).reshape(PIP_H, PIP_W, 4)
            alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
            preview_rgb = (
                preview_rgb.astype(np.float32) * (1.0 - alpha)
                + overlay_rgba[:, :, :3].astype(np.float32) * alpha
            ).astype(np.uint8)
        preview_rgb = np.flipud(preview_rgb).astype(np.uint8, copy=False)
        self.camera_texture.write(preview_rgb.tobytes())
        x = self.width - PIP_W - (PIP_MARGIN + 24)
        y = self.height - PIP_H - (PIP_MARGIN + 28)
        self._render_quad_texture(self.camera_texture, (x, y, PIP_W, PIP_H), opacity=0.94)

    def tick(self) -> float:
        pygame.display.flip()
        self.clock.tick(FPS)
        return self.clock.get_fps()

    def quit(self) -> None:
        self.camera_texture.release()
        self.overlay_texture.release()
        self.quad_vao.release()
        self.quad_buffer.release()
        self.quad_program.release()
        self.point_vao.release()
        self.point_buffer.release()
        self.point_program.release()
        self.ctx.release()
        pygame.quit()
