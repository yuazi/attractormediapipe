from __future__ import annotations

import math
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
    FOG_RANGE,
    FPS,
    LUMINOSITY_RANGE,
    PIP_H,
    PIP_W,
    SCALE_RANGE,
    SPEED_RANGE,
    TRAIL_DECAY,
    WIN_H,
    WIN_W,
)
from hands.skeleton import draw_hand_skeleton

from .background import get_background_layers
from .common import compute_mvp
from .gpu_stepper import TransformFeedbackTrailStepper


FONT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"))


VERTEX_SHADER = """
#version 330

uniform mat4 u_mvp;
uniform float u_time;
uniform float u_point_scale;
uniform vec3 u_center;
uniform float u_inv_extent;
uniform float u_scale_hint;
uniform int u_point_count;

in vec3 in_position;

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
    float age = 1.0;
    if (u_point_count > 1) {
        age = clamp(float(gl_VertexID) / float(u_point_count - 1), 0.02, 1.0);
    }
    vec3 normalized = (in_position - u_center) * u_inv_extent * u_scale_hint;
    float pulse = 1.0 + 0.045 * sin(u_time * 0.85 + age * 4.0);
    float drift = sin(u_time * 0.33) * 0.32;
    vec3 animated = rotationY(drift) * (normalized * pulse);
    vec4 clip = u_mvp * vec4(animated, 1.0);
    gl_Position = clip;

    float depth = max(0.30, clip.w);
    gl_PointSize = clamp((u_point_scale / depth) + 0.8 + age * 2.0, 1.0, 7.5);
    v_age = age;
}
"""


FRAGMENT_SHADER = """
#version 330

uniform vec3 u_color;
uniform float u_luminosity;
uniform float u_trail_decay;
uniform float u_fade;

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
    float decay = mix(u_trail_decay, 1.0, v_age);
    float alpha = glow * mix(0.018, 0.15, v_age) * max(u_luminosity, 0.05) * decay * u_fade;
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
uniform vec2 u_uv_offset;
uniform vec2 u_uv_scale;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 sample_uv = clamp(v_uv * u_uv_scale + u_uv_offset, vec2(0.0), vec2(1.0));
    vec4 color = texture(u_texture, sample_uv);
    fragColor = vec4(color.rgb, color.a * u_opacity);
}
"""

BACKGROUND_FRAGMENT_SHADER = """
#version 330

uniform sampler2D u_base_texture;
uniform sampler2D u_fog_texture;
uniform float u_fog_opacity;
uniform float u_time;
uniform vec2 u_base_uv_offset;
uniform vec2 u_base_uv_scale;
uniform vec2 u_fog_uv_offset;
uniform vec2 u_fog_uv_scale;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 base_uv = v_uv * u_base_uv_scale + u_base_uv_offset;
    vec2 fog_uv = clamp(v_uv * u_fog_uv_scale + u_fog_uv_offset, vec2(0.0), vec2(1.0));
    float organic_1 = sin(base_uv.x * 18.0 + u_time * 0.45) * cos(base_uv.y * 9.0 + u_time * 0.34);
    float organic_2 = cos(base_uv.x * 12.0 - u_time * 0.28) * sin(base_uv.y * 14.0 + u_time * 0.40);
    base_uv += vec2(organic_1 * 0.0020, (organic_1 + organic_2) * 0.0032);
    base_uv = clamp(base_uv, vec2(0.0), vec2(1.0));
    vec4 base_color = texture(u_base_texture, base_uv);
    vec4 fog_color = texture(u_fog_texture, fog_uv);
    float fog_alpha = clamp(fog_color.a * u_fog_opacity, 0.0, 1.0);
    float pulse = 0.95 + 0.05 * sin(u_time * 0.22);
    vec3 rgb = mix(base_color.rgb * pulse, fog_color.rgb, fog_alpha);
    fragColor = vec4(rgb, 1.0);
}
"""

CONTROL_SLIDERS = (
    ("speed", SPEED_RANGE),
    ("fog", FOG_RANGE),
    ("luminosity", LUMINOSITY_RANGE),
    ("scale", SCALE_RANGE),
)
RESET_TRAIL_ACTION = "reset_trail"

HUD_RED = (230, 57, 70)
HUD_RED_DIM = (230, 57, 70)
HUD_GOLD = (244, 162, 97)
HUD_WHITE = (241, 239, 234)
HUD_WHITE_DIM = (241, 239, 234)
HUD_PANEL_BG = (8, 6, 10)
HUD_GESTURE_OK = (82, 183, 136)
HUD_GESTURE_IDLE = (118, 124, 136)
HUD_SCALE = 1.18

HUD_SHORTCUTS = (
    ("1-9", "Switch attractor"),
    ("R", "Reset trail"),
    ("SPACE", "Pause or resume"),
    ("G", "Ghost orbit mode"),
    ("P", "Cycle snapshot preset"),
    ("S", "Export 5K snapshot"),
    ("WHEEL", "Zoom"),
    ("L PINCH", "Speed"),
    ("L RING", "Luminosity"),
    ("L HOLD", "Randomize"),
    ("R PALM", "Yaw and pitch"),
    ("R PINCH", "Scale"),
    ("R RING", "Fog"),
    ("R TAP", "Next study"),
    ("R HOLD", "Previous study"),
)

HUD_THEMES: dict[str, dict[str, object]] = {
    "Lorenz": {"accent": (230, 57, 70), "label": "Lorenz"},
    "Rossler": {"accent": (244, 162, 97), "label": "Rössler"},
    "Halvorsen": {"accent": (82, 183, 136), "label": "Halvorsen"},
    "Dadras": {"accent": (76, 201, 240), "label": "Dadras"},
    "Chen": {"accent": (199, 125, 255), "label": "Chen"},
    "Aizawa": {"accent": (255, 209, 102), "label": "Aizawa"},
    "Thomas": {"accent": (255, 107, 157), "label": "Thomas"},
    "Sprott B": {"accent": (6, 214, 160), "label": "Sprott"},
    "Langford": {"accent": (17, 138, 178), "label": "Langford"},
}


@dataclass(frozen=True)
class HUDLayout:
    nav_rows: tuple[tuple[int, int, int, int], ...]
    params_panel: tuple[int, int, int, int]
    slider_tracks: dict[str, tuple[int, int, int, int]]
    reset_button: tuple[int, int, int, int]
    shortcuts_panel: tuple[int, int, int, int]
    shortcuts_toggle: tuple[int, int, int, int]
    pip_rect: tuple[int, int, int, int]


@dataclass
class SceneState:
    positions: np.ndarray
    ages: np.ndarray
    trail_center: tuple[float, float, float]
    trail_inv_extent: float
    trail_scale_hint: float
    transition_positions: np.ndarray
    transition_center: tuple[float, float, float]
    transition_inv_extent: float
    transition_scale_hint: float
    transition_color: tuple[int, int, int]
    transition_progress: float
    attractor_name: str
    attractor_color: tuple[int, int, int]
    attractor_index: int
    attractor_total: int
    attractor_names: Sequence[str]
    attractor_state: tuple[float, float, float]
    placard_title: str
    placard_year: str
    placard_medium: str
    placard_equation: str
    placard_params: Sequence[tuple[str, str]]
    yaw: float
    pitch: float
    roll: float
    zoom: float
    speed: float
    fog: float
    luminosity: float
    point_count: int
    fps: float
    paused: bool
    ghost_mode: bool
    exporting: bool
    export_message: str
    preset_name: str
    status_message: str
    show_overlay: bool
    show_shortcuts: bool
    show_camera: bool
    focus_mode: bool
    active_slider: Optional[str]
    hover_slider: Optional[str]
    hover_nav_index: Optional[int]
    hover_reset: bool
    hover_shortcuts_toggle: bool
    hover_pip: bool
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
        self._hud_scale = HUD_SCALE

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
        self.point_vao = self.ctx.vertex_array(self.point_program, [(self.point_buffer, "3f", "in_position")])
        self.gpu_stepper = TransformFeedbackTrailStepper.create_if_supported(self.ctx)

        self.quad_program = self.ctx.program(vertex_shader=QUAD_VERTEX_SHADER, fragment_shader=QUAD_FRAGMENT_SHADER)
        self.background_program = self.ctx.program(vertex_shader=QUAD_VERTEX_SHADER, fragment_shader=BACKGROUND_FRAGMENT_SHADER)
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
        self.background_vao = self.ctx.vertex_array(
            self.background_program,
            [(self.quad_buffer, "2f 2f", "in_pos", "in_uv")],
        )

        self.background_texture = self.ctx.texture(self._overlay_size, 4)
        self.background_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fog_texture = self.ctx.texture(self._overlay_size, 4)
        self.fog_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.overlay_texture = self.ctx.texture(self._overlay_size, 4)
        self.overlay_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.camera_texture = self.ctx.texture((PIP_W, PIP_H), 3)
        self.camera_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._font_mono_7 = self._load_font(self._s(7), "hud_mono")
        self._font_mono_8 = self._load_font(self._s(8), "hud_mono")
        self._font_mono_9 = self._load_font(self._s(9), "hud_mono")
        self._font_mono_10 = self._load_font(self._s(10), "hud_mono")
        self._font_mono_11 = self._load_font(self._s(11), "hud_mono")
        self._font_mono_18 = self._load_font(self._s(18), "hud_mono")
        self._font_body_9 = self._load_font(self._s(9), "hud_body")
        self._font_body_10 = self._load_font(self._s(10), "hud_body")
        self._font_body_11 = self._load_font(self._s(11), "hud_body")
        self._font_body_11_medium = self._load_font(self._s(11), "hud_body_medium")
        self._font_body_13 = self._load_font(self._s(13), "hud_body_medium")
        self._font_display_13 = self._load_font(self._s(13), "hud_display")
        self._font_display_32 = self._load_font(self._s(32), "hud_display")
        self.clock = pygame.time.Clock()
        self._background_time = 0.0
        self._shortcuts_progress = 1.0
        self._live_axis = 0
        self._last_state_name: str | None = None
        self._last_state_vector: tuple[float, float, float] | None = None
        self._fps_history: list[float] = [30.0] * 5
        self._background_layer_key: tuple[int, int, tuple[int, int, int]] | None = None
        self._fog_layer_key: tuple[int, int, tuple[int, int, int]] | None = None
        self._vignette_cache: dict[tuple[int, int], Image.Image] = {}
        self._scanlines_cache: dict[tuple[int, int], Image.Image] = {}

    def _load_font(self, size: int, style: str):
        families = {
            "hud_mono": [
                os.path.join(FONT_DIR, "Onest-Medium.ttf"),
                os.path.join(FONT_DIR, "Onest-Regular.ttf"),
                os.path.join(FONT_DIR, "PlusJakartaSans-VariableFont_wght.ttf"),
                "~/Library/Fonts/Onest-Medium.ttf",
                "~/Library/Fonts/Onest-Regular.ttf",
                "~/Library/Fonts/PlusJakartaSans-VariableFont_wght.ttf",
                "~/Library/Fonts/neuehaasgrottext-55roman-trial.otf",
                "DejaVuSans.ttf",
            ],
            "hud_body": [
                os.path.join(FONT_DIR, "PlusJakartaSans-VariableFont_wght.ttf"),
                os.path.join(FONT_DIR, "Onest-Regular.ttf"),
                "~/Library/Fonts/PlusJakartaSans-VariableFont_wght.ttf",
                "~/Library/Fonts/Onest-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "DejaVuSans.ttf",
            ],
            "hud_body_medium": [
                os.path.join(FONT_DIR, "Onest-SemiBold.ttf"),
                os.path.join(FONT_DIR, "Onest-Medium.ttf"),
                os.path.join(FONT_DIR, "PlusJakartaSans-VariableFont_wght.ttf"),
                "~/Library/Fonts/Onest-SemiBold.ttf",
                "~/Library/Fonts/Onest-Bold.ttf",
                "~/Library/Fonts/PlusJakartaSans-VariableFont_wght.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "DejaVuSans-Bold.ttf",
            ],
            "hud_display": [
                os.path.join(FONT_DIR, "BebasNeue-Regular.ttf"),
                "~/Library/Fonts/BebasNeue-Regular.ttf",
                "~/Library/Fonts/neuehaasgrotdisp-65medium-trial.otf",
                "/System/Library/Fonts/Supplemental/Impact.ttf",
                "DejaVuSans-Bold.ttf",
            ],
        }
        for candidate in families.get(style, families["hud_body"]):
            candidate_path = os.path.expanduser(candidate)
            if os.path.isabs(candidate_path) and not os.path.exists(candidate_path):
                continue
            try:
                return ImageFont.truetype(candidate_path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _s(self, value: float, *, minimum: int = 1) -> int:
        return max(minimum, int(round(value * self._hud_scale)))

    def _sync_window_size(self) -> None:
        width, height = pygame.display.get_window_size()
        if (width, height) != (self.width, self.height):
            self.width, self.height = width, height
        self.ctx.viewport = (0, 0, self.width, self.height)
        if (self.width, self.height) != self._overlay_size:
            self.background_texture.release()
            self.fog_texture.release()
            self.overlay_texture.release()
            self._overlay_size = (self.width, self.height)
            self.background_texture = self.ctx.texture(self._overlay_size, 4)
            self.background_texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
            self.fog_texture = self.ctx.texture(self._overlay_size, 4)
            self.fog_texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
            self.overlay_texture = self.ctx.texture(self._overlay_size, 4)
            self.overlay_texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
            self._background_layer_key = None
            self._fog_layer_key = None

    def _ensure_point_capacity(self, byte_count: int) -> None:
        if byte_count <= self._point_capacity:
            return
        self._point_capacity = max(byte_count, self._point_capacity * 2)
        self.point_buffer.release()
        self.point_buffer = self.ctx.buffer(reserve=self._point_capacity)
        self.point_vao.release()
        self.point_vao = self.ctx.vertex_array(self.point_program, [(self.point_buffer, "3f", "in_position")])

    def _render_quad_texture(
        self,
        texture,
        rect: tuple[int, int, int, int],
        opacity: float = 1.0,
        *,
        uv_offset: tuple[float, float] = (0.0, 0.0),
        uv_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.ctx.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE_MINUS_SRC_ALPHA
        texture.use(location=0)
        self.quad_program["u_texture"].value = 0
        self.quad_program["u_opacity"].value = float(opacity)
        self.quad_program["u_uv_offset"].value = tuple(float(value) for value in uv_offset)
        self.quad_program["u_uv_scale"].value = tuple(float(value) for value in uv_scale)
        self.quad_program["u_screen_size"].value = (float(self.width), float(self.height))
        self.quad_program["u_rect"].value = tuple(float(value) for value in rect)
        self.quad_vao.render()

    def _render_background_composite(
        self,
        rect: tuple[int, int, int, int],
        *,
        time_value: float,
        fog_opacity: float,
        base_offset: tuple[float, float],
        base_scale: tuple[float, float],
        fog_offset: tuple[float, float],
        fog_scale: tuple[float, float],
    ) -> None:
        self.ctx.blend_func = self.moderngl.SRC_ALPHA, self.moderngl.ONE_MINUS_SRC_ALPHA
        self.background_texture.use(location=0)
        self.fog_texture.use(location=1)
        self.background_program["u_base_texture"].value = 0
        self.background_program["u_fog_texture"].value = 1
        self.background_program["u_fog_opacity"].value = float(max(0.0, min(1.0, fog_opacity)))
        self.background_program["u_base_uv_offset"].value = tuple(float(value) for value in base_offset)
        self.background_program["u_base_uv_scale"].value = tuple(float(value) for value in base_scale)
        self.background_program["u_fog_uv_offset"].value = tuple(float(value) for value in fog_offset)
        self.background_program["u_fog_uv_scale"].value = tuple(float(value) for value in fog_scale)
        self.background_program["u_time"].value = float(time_value)
        self.background_program["u_screen_size"].value = (float(self.width), float(self.height))
        self.background_program["u_rect"].value = tuple(float(value) for value in rect)
        self.background_vao.render()

    def _render_trail_points(
        self,
        positions: np.ndarray,
        *,
        center: tuple[float, float, float],
        inv_extent: float,
        scale_hint: float,
        color: tuple[int, int, int],
        luminosity: float,
        time_value: float,
        fade: float,
        mvp: np.ndarray,
    ) -> None:
        if len(positions) == 0 or fade <= 0.0:
            return

        vertex_data = np.asarray(positions, dtype=np.float32)
        self._ensure_point_capacity(vertex_data.nbytes)
        self.point_buffer.write(vertex_data.tobytes())
        self.point_program["u_mvp"].write(mvp.T.astype(np.float32, copy=False).tobytes())
        self.point_program["u_time"].value = float(time_value)
        self.point_program["u_point_scale"].value = float(DEFAULT_POINT_SIZE)
        self.point_program["u_luminosity"].value = float(luminosity)
        self.point_program["u_color"].value = tuple(channel / 255.0 for channel in color)
        self.point_program["u_center"].value = tuple(float(value) for value in center)
        self.point_program["u_inv_extent"].value = float(inv_extent)
        self.point_program["u_scale_hint"].value = float(scale_hint)
        self.point_program["u_point_count"].value = int(len(vertex_data))
        self.point_program["u_trail_decay"].value = float(TRAIL_DECAY)
        self.point_program["u_fade"].value = float(max(0.0, min(1.0, fade)))
        self.ctx.blend_func = self.moderngl.ONE, self.moderngl.ONE
        self.point_vao.render(mode=self.moderngl.POINTS, vertices=len(vertex_data))

    def draw(self, state: SceneState) -> None:
        self._sync_window_size()
        accent = self._hud_accent(state.attractor_name, state.attractor_color)
        layout = self._hud_layout(state.attractor_names)
        background = tuple(channel / 255.0 for channel in BACKGROUND_COLOR)
        self.ctx.clear(background[0], background[1], background[2], 1.0)
        self._background_time = float(state.time_value)
        self._render_atmosphere(accent, state.fog)

        if state.fps > 0.0:
            self._fps_history.append(state.fps)
            self._fps_history = self._fps_history[-5:]

        mvp = compute_mvp(
            self.width,
            self.height,
            yaw=state.yaw,
            pitch=state.pitch,
            roll=state.roll,
            zoom=state.zoom,
        )
        if len(state.transition_positions) > 0 and state.transition_progress < 1.0:
            self._render_trail_points(
                state.transition_positions,
                center=state.transition_center,
                inv_extent=state.transition_inv_extent,
                scale_hint=state.transition_scale_hint,
                color=state.transition_color,
                luminosity=state.luminosity,
                time_value=state.time_value,
                fade=1.0 - state.transition_progress,
                mvp=mvp,
            )
        self._render_trail_points(
            state.positions,
            center=state.trail_center,
            inv_extent=state.trail_inv_extent,
            scale_hint=state.trail_scale_hint,
            color=state.attractor_color,
            luminosity=state.luminosity,
            time_value=state.time_value,
            fade=state.transition_progress,
            mvp=mvp,
        )

        if state.show_overlay and not state.focus_mode and state.show_camera and state.pip_frame is not None:
            self._draw_camera_frame(
                state.pip_frame,
                state.left_landmarks,
                state.right_landmarks,
                state.left_pip_caption,
                state.right_pip_caption,
                rect=layout.pip_rect,
                opacity=0.94,
            )
        if state.show_overlay and not state.focus_mode:
            self._draw_overlay(state, layout, accent)
        elif (state.show_camera or state.focus_mode) and state.pip_frame is not None:
            self._draw_camera_frame(
                state.pip_frame,
                state.left_landmarks,
                state.right_landmarks,
                state.left_pip_caption,
                state.right_pip_caption,
                rect=layout.pip_rect,
            )
        if state.focus_mode:
            self._draw_focus_status_bar(state, accent)

    def _render_atmosphere(self, accent: tuple[int, int, int], fog: float) -> None:
        cache_key = (self.width, self.height, accent)
        time_value = float(getattr(self, "_background_time", 0.0))
        if self._background_layer_key != cache_key or self._fog_layer_key != cache_key:
            layers = get_background_layers(self.width, self.height, accent)
            if self._background_layer_key != cache_key:
                self.background_texture.write(layers.texture_rgba.tobytes())
                self._background_layer_key = cache_key
            if self._fog_layer_key != cache_key:
                self.fog_texture.write(layers.fog_rgba.tobytes())
                self._fog_layer_key = cache_key

        base_scale = (0.965, 0.965)
        base_offset = (
            0.012 + math.sin(time_value * 0.032) * 0.008,
            0.018 + math.cos(time_value * 0.029) * 0.010,
        )
        fog_offset = (
            0.026 + math.cos(time_value * 0.018 + 0.8) * 0.014,
            0.032 + math.sin(time_value * 0.016 + 0.2) * 0.014,
        )
        self._render_background_composite(
            (0, 0, self.width, self.height),
            time_value=time_value,
            fog_opacity=fog * (0.94 + 0.06 * math.sin(time_value * 0.18 + 0.3)),
            base_offset=base_offset,
            base_scale=base_scale,
            fog_offset=fog_offset,
            fog_scale=(0.93, 0.93),
        )

    def _draw_overlay(self, state: SceneState, layout: HUDLayout, accent: tuple[int, int, int]) -> None:
        image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        target_progress = 1.0 if state.show_shortcuts else 0.0
        self._shortcuts_progress += (target_progress - self._shortcuts_progress) * 0.24
        if abs(target_progress - self._shortcuts_progress) < 0.01:
            self._shortcuts_progress = target_progress

        self._update_live_axis(state)
        self._draw_coordinates(draw, state, accent)
        self._draw_title_bar(draw, accent)
        self._draw_navigation(draw, state, layout, accent)
        self._draw_shortcuts_panel(draw, state, layout, accent)
        self._draw_params_panel(draw, state, layout, accent)
        self._draw_study_info(draw, state, accent)
        self._draw_status_bar(draw, state, accent)
        self._draw_pip_panel(draw, state, layout, accent)
        image.alpha_composite(self._scanline_layer())
        image.alpha_composite(self._vignette_layer())
        draw = ImageDraw.Draw(image)
        self._draw_corner_fiducials(draw)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.overlay_texture.write(image.tobytes())
        self._render_quad_texture(self.overlay_texture, (0, 0, self.width, self.height))

    def _update_live_axis(self, state: SceneState) -> None:
        if self._last_state_name != state.attractor_name or self._last_state_vector is None:
            self._live_axis = int(np.argmax(np.abs(np.asarray(state.attractor_state, dtype=np.float32))))
        else:
            deltas = [
                abs(float(state.attractor_state[index]) - float(self._last_state_vector[index]))
                for index in range(3)
            ]
            if max(deltas) > 1e-5:
                self._live_axis = int(np.argmax(deltas))
        self._last_state_name = state.attractor_name
        self._last_state_vector = tuple(float(value) for value in state.attractor_state)

    def _hud_layout(self, attractor_names: Sequence[str]) -> HUDLayout:
        s = self._s
        pad_x = s(28)
        pad_top = s(28)
        pad_bottom = s(24)
        usable_width = self.width - pad_x * 2
        usable_height = self.height - pad_top - pad_bottom
        col_width = usable_width / 3.0
        row_height = usable_height / 3.0

        nav_rows: list[tuple[int, int, int, int]] = []
        nav_right = self.width - pad_x
        nav_top = pad_top + s(24)
        nav_spacing = s(22)
        for index, name in enumerate(attractor_names):
            label = self._hud_label(name)
            row_width = s(38) + self._text_width(label, self._font_mono_10) + s(28)
            row_left = nav_right - row_width
            row_top = nav_top + index * nav_spacing
            nav_rows.append((row_left, row_top, nav_right, row_top + s(18)))

        params_width = max(s(236), min(s(258), int(col_width) - s(12)))
        params_height = s(278)
        params_left = self.width - pad_x - params_width
        params_top = int(pad_top + row_height + max(s(12), (row_height - params_height) / 2.0))
        slider_tracks: dict[str, tuple[int, int, int, int]] = {}
        row_top = params_top + s(46)
        for slider_id, _value_range in CONTROL_SLIDERS:
            slider_tracks[slider_id] = (
                params_left + s(18),
                row_top + s(22),
                params_left + params_width - s(18),
                row_top + s(28),
            )
            row_top += s(48)
        reset_button = (
            params_left + s(18),
            params_top + params_height - s(42),
            params_left + params_width - s(18),
            params_top + params_height - s(12),
        )

        shortcuts_width = max(s(252), min(s(282), int(col_width) - s(10)))
        shortcuts_height = s(232)
        shortcuts_left = pad_x
        shortcuts_top = int(pad_top + row_height + max(s(16), (row_height - shortcuts_height) / 2.0))
        shortcuts_panel = (shortcuts_left, shortcuts_top, shortcuts_left + shortcuts_width, shortcuts_top + shortcuts_height)
        shortcuts_toggle = (shortcuts_panel[2] - s(62), shortcuts_top + s(8), shortcuts_panel[2] - s(10), shortcuts_top + s(24))

        pip_rect = (self.width - pad_x - PIP_W, self.height - pad_bottom - PIP_H, self.width - pad_x, self.height - pad_bottom)
        return HUDLayout(
            nav_rows=tuple(nav_rows),
            params_panel=(params_left, params_top, params_left + params_width, params_top + params_height),
            slider_tracks=slider_tracks,
            reset_button=reset_button,
            shortcuts_panel=shortcuts_panel,
            shortcuts_toggle=shortcuts_toggle,
            pip_rect=pip_rect,
        )

    def _hud_accent(self, attractor_name: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
        theme = HUD_THEMES.get(attractor_name, {})
        return tuple(theme.get("accent", fallback))  # type: ignore[arg-type]

    def _hud_label(self, attractor_name: str) -> str:
        theme = HUD_THEMES.get(attractor_name, {})
        return str(theme.get("label", attractor_name))

    def _draw_coordinates(self, draw: ImageDraw.ImageDraw, state: SceneState, accent: tuple[int, int, int]) -> None:
        s = self._s
        x = s(28)
        y = s(28)
        draw.text((x, y), "CURRENT POSITION", font=self._font_mono_8, fill=self._rgba(accent, 255))
        axis_y = y + s(16)
        value_y = axis_y + s(12)
        gap = s(82)
        blink_on = int(state.time_value / 0.55) % 2 == 0
        for index, axis in enumerate(("X", "Y", "Z")):
            axis_color = self._rgba(accent, 240 if index == self._live_axis else 140)
            draw.text((x, axis_y), axis, font=self._font_mono_8, fill=axis_color)
            value = self._format_coordinate(state.attractor_state[index])
            draw.text((x, value_y), value, font=self._font_mono_18, fill=self._rgba(HUD_WHITE, 255))
            if index == self._live_axis and blink_on:
                caret_x = x + self._text_width(value, self._font_mono_18) + s(2)
                draw.text((caret_x, value_y - s(1)), "|", font=self._font_mono_18, fill=self._rgba(accent, 255))
            x += gap

    def _draw_title_bar(self, draw: ImageDraw.ImageDraw, accent: tuple[int, int, int]) -> None:
        s = self._s
        app_label = "(y)us"
        title = "PARTICLE ATTRACTOR"
        app_width = self._text_width(app_label, self._font_mono_8)
        title_width = self._text_width(title, self._font_display_13)
        center_x = self.width // 2
        top = s(28)
        draw.text((center_x - app_width // 2, top), app_label, font=self._font_mono_8, fill=self._rgba(accent, 76))
        draw.text((center_x - title_width // 2, top + s(12)), title, font=self._font_display_13, fill=self._rgba(accent, 148))

    def _draw_navigation(self, draw: ImageDraw.ImageDraw, state: SceneState, layout: HUDLayout, accent: tuple[int, int, int]) -> None:
        s = self._s
        header = "STUDIES"
        header_width = self._text_width(header, self._font_mono_8)
        draw.text((self.width - s(28) - header_width, s(28)), header, font=self._font_mono_8, fill=self._rgba(accent, 196))

        for index, (name, row_rect) in enumerate(zip(state.attractor_names, layout.nav_rows)):
            left, top, right, bottom = row_rect
            active = index == state.attractor_index
            hovered = index == state.hover_nav_index
            accent = self._hud_accent(name, state.attractor_color)
            label = self._hud_label(name)
            if active:
                draw.rounded_rectangle((left, top, right, bottom), radius=s(5), fill=self._rgba(accent, 34))
                draw.line((left, top, left, bottom), fill=self._rgba(accent, 255), width=max(1, s(2)))
            elif hovered:
                draw.rounded_rectangle((left, top, right, bottom), radius=s(5), fill=self._rgba(accent, 18))

            stripe_alpha = 255 if active or hovered else 90
            draw.rounded_rectangle((left + s(8), top + s(4), left + s(10), top + s(14)), radius=s(1), fill=self._rgba(accent, stripe_alpha))
            num = f"{index + 1:02d}"
            draw.text((left + s(18), top + s(3)), num, font=self._font_mono_8, fill=self._rgba(accent if active or hovered else HUD_WHITE, 255 if active or hovered else 72))
            label_width = self._text_width(label, self._font_mono_10)
            label_x = right - s(20) - label_width
            label_fill = self._rgba(HUD_WHITE, 255 if active or hovered else 120)
            draw.text((label_x, top + s(1)), label, font=self._font_mono_10, fill=label_fill)

            dot_center_x = right - s(8)
            dot_center_y = top + s(9)
            if active:
                self._draw_dot(draw, (dot_center_x, dot_center_y), max(2, s(2)), accent, glow=s(10))
            elif hovered:
                self._draw_dot(draw, (dot_center_x, dot_center_y), max(2, s(2)), accent, glow=s(5), alpha=140)

    def _draw_shortcuts_panel(
        self,
        draw: ImageDraw.ImageDraw,
        state: SceneState,
        layout: HUDLayout,
        accent: tuple[int, int, int],
    ) -> None:
        s = self._s
        left, top, right, bottom = layout.shortcuts_panel
        header_height = s(34)
        full_height = bottom - top
        current_height = header_height + int((full_height - header_height) * self._shortcuts_progress)
        panel_bottom = top + current_height

        draw.rounded_rectangle((left, top, right, panel_bottom), radius=s(14), fill=self._rgba(HUD_PANEL_BG, 198), outline=self._rgba(accent, 40), width=1)
        draw.line((left, top + 1, left, panel_bottom - 1), fill=self._rgba(accent, 255), width=max(1, s(2)))
        draw.text((left + s(14), top + s(10)), "SHORTCUTS", font=self._font_mono_8, fill=self._rgba(accent, 255))

        toggle_label = "[H] HIDE" if state.show_shortcuts else "[H] SHOW"
        toggle_left, toggle_top, toggle_right, toggle_bottom = layout.shortcuts_toggle
        if state.hover_shortcuts_toggle:
            draw.rounded_rectangle((toggle_left, toggle_top, toggle_right, toggle_bottom), radius=s(5), fill=self._rgba(accent, 18), outline=self._rgba(accent, 130), width=1)
        else:
            draw.rounded_rectangle((toggle_left, toggle_top, toggle_right, toggle_bottom), radius=s(5), fill=self._rgba(HUD_PANEL_BG, 0), outline=self._rgba(accent, 40), width=1)
        toggle_width = self._text_width(toggle_label, self._font_mono_7)
        draw.text((toggle_left + ((toggle_right - toggle_left - toggle_width) // 2), toggle_top + s(4)), toggle_label, font=self._font_mono_7, fill=self._rgba(accent, 148))

        if self._shortcuts_progress <= 0.03:
            return

        clip_bottom = panel_bottom - s(10)
        key_x = left + s(14)
        cursor_y = top + s(40)
        row_height = s(18)
        row_alpha = max(0, min(255, int(255 * self._shortcuts_progress)))
        chip_widths = [max(s(28), self._text_width(key, self._font_mono_8) + s(12)) for key, _ in HUD_SHORTCUTS]
        chip_col_width = max(chip_widths, default=s(28))
        desc_x = key_x + chip_col_width + s(14)
        desc_width = max(s(80), right - s(16) - desc_x)
        for key, description in HUD_SHORTCUTS:
            if cursor_y + s(10) > clip_bottom:
                break
            chip_width = max(s(28), self._text_width(key, self._font_mono_8) + s(12))
            chip_left = key_x + max(0, (chip_col_width - chip_width) // 2)
            draw.rounded_rectangle(
                (chip_left, cursor_y - s(1), chip_left + chip_width, cursor_y + s(10)),
                radius=s(4),
                fill=self._rgba(HUD_WHITE, 12),
                outline=self._rgba(accent, 40),
                width=1,
            )
            draw.text((chip_left + s(6), cursor_y), key, font=self._font_mono_8, fill=self._rgba(HUD_WHITE, row_alpha))
            description_lines = self._wrap_text_pixels(description, desc_width, self._font_body_10)
            for line_index, line in enumerate(description_lines[:2]):
                line_y = cursor_y + line_index * s(9)
                if line_y + s(8) > clip_bottom:
                    break
                draw.text((desc_x, line_y), line, font=self._font_body_10, fill=self._rgba(HUD_WHITE, min(row_alpha, 152)))
            cursor_y += row_height

    def _draw_params_panel(self, draw: ImageDraw.ImageDraw, state: SceneState, layout: HUDLayout, accent: tuple[int, int, int]) -> None:
        s = self._s
        left, top, right, bottom = layout.params_panel
        draw.rounded_rectangle((left, top, right, bottom), radius=s(14), fill=self._rgba(HUD_PANEL_BG, 199), outline=self._rgba(accent, 40), width=1)
        draw.line((left + 1, top, right - 1, top), fill=self._rgba(accent, 255), width=1)
        draw.text((left + s(18), top + s(12)), "PARAMETERS", font=self._font_mono_9, fill=self._rgba(accent, 188))

        live_label = "EXPORT" if state.exporting else "GHOST" if state.ghost_mode else "PAUSED" if state.paused else "LIVE"
        live_x = right - s(76)
        live_y = top + s(12)
        self._draw_dot(draw, (live_x, live_y + s(5)), max(2, s(2)), accent, glow=s(8))
        draw.text((live_x + s(8), live_y + s(1)), live_label, font=self._font_mono_8, fill=self._rgba(accent, 240))

        rows = {
            "speed": ("Speed", state.speed, SPEED_RANGE, accent),
            "fog": ("Fog", state.fog, FOG_RANGE, accent),
            "luminosity": ("Luminosity", state.luminosity, LUMINOSITY_RANGE, accent),
            "scale": ("Scale", state.zoom, SCALE_RANGE, accent),
        }
        for slider_id, _value_range in CONTROL_SLIDERS:
            label, value, value_range, fill_color = rows[slider_id]
            track_left, track_top, track_right, _track_bottom = layout.slider_tracks[slider_id]
            self._draw_slider_row(
                draw,
                track_left,
                track_top - s(20),
                track_right - track_left,
                label,
                float(value),
                value_range,
                fill_color,
                active=state.active_slider == slider_id or state.hover_slider == slider_id,
            )

        divider_y = layout.reset_button[1] - s(10)
        draw.line((left + s(18), divider_y, right - s(18), divider_y), fill=self._rgba(accent, 22), width=1)
        button_left, button_top, button_right, button_bottom = layout.reset_button
        if state.hover_reset:
            draw.rounded_rectangle((button_left, button_top, button_right, button_bottom), radius=s(10), fill=self._rgba(accent, 24), outline=self._rgba(accent, 200), width=1)
        else:
            draw.rounded_rectangle((button_left, button_top, button_right, button_bottom), radius=s(10), fill=self._rgba(HUD_PANEL_BG, 0), outline=self._rgba(accent, 140), width=1)
        button_label = "RESET TRAIL"
        button_width = self._text_width(button_label, self._font_mono_9)
        draw.text((button_left + ((button_right - button_left - button_width) // 2), button_top + s(9)), button_label, font=self._font_mono_9, fill=self._rgba(accent if not state.hover_reset else HUD_WHITE, 255))

    def _draw_slider_row(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        value_range: tuple[float, float],
        fill_color: tuple[int, int, int],
        *,
        active: bool,
    ) -> None:
        s = self._s
        minimum, maximum = value_range
        ratio = 0.0 if maximum <= minimum else max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
        track_y = y + s(24)
        value_text = f"{value:0.2f}"
        draw.text((x, y), label.upper(), font=self._font_mono_8, fill=self._rgba(HUD_WHITE, 140))
        value_width = self._text_width(value_text, self._font_mono_11)
        draw.text((x + width - value_width, y - s(1)), value_text, font=self._font_mono_11, fill=self._rgba(HUD_WHITE, 255))
        draw.line((x, track_y, x + width, track_y), fill=self._rgba(HUD_WHITE, 22), width=max(1, s(2)))
        fill_end = x + int(width * ratio)
        if fill_end > x:
            draw.line((x, track_y, fill_end, track_y), fill=self._rgba(fill_color, 255), width=max(1, s(2)))
        thumb_radius = s(5)
        if active:
            self._draw_dot(draw, (fill_end, track_y), thumb_radius, HUD_WHITE, glow=s(14), alpha=255)
        else:
            draw.ellipse((fill_end - thumb_radius, track_y - thumb_radius, fill_end + thumb_radius, track_y + thumb_radius), fill=self._rgba(HUD_WHITE, 235))

    def _draw_study_info(self, draw: ImageDraw.ImageDraw, state: SceneState, accent: tuple[int, int, int]) -> None:
        s = self._s
        left = s(28)
        placard_width = s(360)
        title_lines = self._study_title_lines(state.placard_title)
        rows = list(state.placard_params[:3]) + [("points", f"{state.point_count:,}")]
        row_label_width = max((self._text_width(label.upper(), self._font_mono_8) for label, _ in rows), default=0)
        row_value_width = max((self._text_width(value, self._font_mono_8) for _, value in rows), default=0)
        column_gap = s(24)
        stats_gap = s(16)
        stats_width = max(s(120), row_label_width + stats_gap + row_value_width)
        body_width = max(s(180), placard_width - column_gap - stats_width)
        equation_lines = self._format_equation_lines(state.placard_equation, body_width)
        description_lines = self._wrap_text_pixels(state.placard_medium, body_width, self._font_body_10)
        title_height = len(title_lines) * s(28)
        body_block_height = s(18) + len(equation_lines) * s(12) + s(8) + len(description_lines) * s(14)
        stats_block_height = len(rows) * s(13)
        content_height = s(18) + title_height + s(10) + max(body_block_height, stats_block_height)
        top = self.height - s(24) - content_height

        draw.line((left, top, left + s(22), top), fill=self._rgba(accent, 160), width=1)
        draw.text((left + s(32), top - s(4)), f"STUDY NO. {self._roman(state.attractor_index + 1)}", font=self._font_mono_8, fill=self._rgba(accent, 180))

        cursor_y = top + s(12)
        for line in title_lines:
            draw.text((left, cursor_y), line, font=self._font_display_32, fill=self._rgba(accent, 255))
            cursor_y += s(28)

        body_top = cursor_y + s(2)
        body_left = left
        stats_left = left + body_width + column_gap

        draw.text((body_left, body_top), state.placard_year, font=self._font_body_11, fill=self._rgba(HUD_WHITE, 118))
        cursor_y = body_top + s(18)
        for line in equation_lines:
            draw.text((body_left, cursor_y), line, font=self._font_mono_9, fill=self._rgba(HUD_WHITE, 112))
            cursor_y += s(12)
        cursor_y += s(6)
        for line in description_lines:
            draw.text((body_left, cursor_y), line, font=self._font_body_10, fill=self._rgba(HUD_WHITE, 102))
            cursor_y += s(14)

        stats_y = body_top + s(1)
        for label, value in rows:
            draw.text((stats_left, stats_y), label.upper(), font=self._font_mono_8, fill=self._rgba(accent, 132))
            value_width = self._text_width(value, self._font_mono_8)
            draw.text((left + placard_width - value_width, stats_y), value, font=self._font_mono_8, fill=self._rgba(HUD_WHITE, 160))
            stats_y += s(13)

    def _draw_status_bar(self, draw: ImageDraw.ImageDraw, state: SceneState, accent: tuple[int, int, int]) -> None:
        s = self._s
        bar_width = max(2, s(3))
        gap = s(2)
        heights: list[int] = []
        for fps_value in self._fps_history[-5:]:
            clamped = max(0.0, min(60.0, fps_value))
            heights.append(s(5) + int((clamped / 60.0) * s(7)))
        while len(heights) < 5:
            heights.insert(0, s(6))

        total_width = len(heights) * bar_width + (len(heights) - 1) * gap
        base_x = self.width // 2 - s(70)
        base_y = self.height - s(30)
        for index, height in enumerate(heights):
            x = base_x + index * (bar_width + gap)
            y = base_y - height
            draw.rounded_rectangle((x, y, x + bar_width, base_y), radius=s(1), fill=self._rgba(accent, 210 if index >= len(heights) - 3 else 132))

        fps_label = f"{state.fps:0.1f} FPS"
        draw.text((base_x + total_width + s(10), base_y - s(10)), fps_label, font=self._font_mono_10, fill=self._rgba(HUD_WHITE, 220))
        status_text = f"{state.point_count:,} PTS"
        if state.exporting:
            status_text = "EXPORTING"
        elif state.ghost_mode:
            status_text = "GHOST"
        elif state.paused:
            status_text = "PAUSED"
        elif state.status_message:
            status_text = state.status_message
        draw.text((base_x + total_width + s(78), base_y - s(10)), status_text, font=self._font_mono_8, fill=self._rgba(accent, 140))

        preset_label = state.preset_name.upper()
        preset_width = self._text_width(preset_label, self._font_mono_8)
        draw.text((self.width - s(28) - preset_width, base_y - s(10)), preset_label, font=self._font_mono_8, fill=self._rgba(HUD_WHITE, 160))

    def _draw_focus_status_bar(self, state: SceneState, accent: tuple[int, int, int]) -> None:
        image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        s = self._s
        bar_height = s(26)
        top = self.height - bar_height
        draw.rectangle((0, top, self.width, self.height), fill=self._rgba(HUD_PANEL_BG, 168))
        draw.line((0, top, self.width, top), fill=self._rgba(accent, 144), width=1)

        parts = [
            self._hud_label(state.attractor_name).upper(),
            f"PRESET {state.preset_name.upper()}",
            f"{state.fps:0.1f} FPS",
        ]
        if state.ghost_mode:
            parts.append("GHOST")
        if state.status_message:
            parts.append(state.status_message)
        label = "  ·  ".join(parts)
        draw.text((s(18), top + s(8)), label, font=self._font_mono_9, fill=self._rgba(HUD_WHITE, 226))

        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.overlay_texture.write(image.tobytes())
        self._render_quad_texture(self.overlay_texture, (0, 0, self.width, self.height))

    def _draw_pip_panel(self, draw: ImageDraw.ImageDraw, state: SceneState, layout: HUDLayout, accent: tuple[int, int, int]) -> None:
        s = self._s
        left, top, right, bottom = layout.pip_rect
        has_live_camera = state.show_camera and state.pip_frame is not None
        border_color = self._rgba(accent, 88 if not state.hover_pip else 132)
        overlay_alpha = 158 if not has_live_camera else 6
        draw.rounded_rectangle((left, top, right, bottom), radius=s(8), fill=self._rgba(HUD_PANEL_BG, overlay_alpha), outline=border_color, width=1)
        draw.line((left + 1, top, right - 1, top), fill=self._rgba(accent, 120 if not state.hover_pip else 160), width=1)

        if not has_live_camera:
            center_x = left + (right - left) * 0.42
            center_y = top + (bottom - top) * 0.66
            draw.ellipse((center_x - s(46), center_y - s(34), center_x + s(46), center_y + s(34)), fill=self._rgba(HUD_WHITE, 12))
            draw.ellipse((center_x - s(24), center_y - s(18), center_x + s(24), center_y + s(18)), fill=self._rgba(accent, 10))

        scanline_alpha = 6 if has_live_camera else 14
        for scan_y in range(top + 1, bottom, s(3)):
            draw.line((left + 1, scan_y, right - 1, scan_y), fill=self._rgba((0, 0, 0), scanline_alpha), width=1)

        bracket_length = s(8)
        bracket_alpha = 240
        for start_x, start_y, dir_x, dir_y in (
            (left, top, 1, 1),
            (right, top, -1, 1),
            (left, bottom, 1, -1),
            (right, bottom, -1, -1),
        ):
            draw.line((start_x, start_y, start_x + bracket_length * dir_x, start_y), fill=self._rgba(accent, bracket_alpha), width=1)
            draw.line((start_x, start_y, start_x, start_y + bracket_length * dir_y), fill=self._rgba(accent, bracket_alpha), width=1)

        top_label = "CAM · 01"
        draw.text((left + s(7), top + s(6)), top_label, font=self._font_mono_7, fill=self._rgba(HUD_WHITE, 92))
        self._draw_dot(draw, (right - s(32), top + s(9)), max(2, s(2)), accent, glow=s(6))
        draw.text((right - s(24), top + s(5)), "REC", font=self._font_mono_7, fill=self._rgba(accent, 255))

        gesture_live = state.left_detected or state.right_detected
        gesture_label = "GESTURE ACTIVE" if gesture_live else "GESTURE IDLE"
        gesture_color = HUD_GESTURE_OK if gesture_live else HUD_GESTURE_IDLE
        draw.text((left + s(7), bottom - s(12)), self._format_timestamp(state.time_value), font=self._font_mono_8, fill=self._rgba(HUD_WHITE, 92))
        self._draw_dot(draw, (right - s(69), bottom - s(9)), max(2, s(2)), gesture_color, glow=s(6) if gesture_live else s(3), alpha=220 if gesture_live else 140)
        draw.text((right - s(63), bottom - s(13)), gesture_label, font=self._font_mono_7, fill=self._rgba(gesture_color, 220 if gesture_live else 132))

    def _draw_corner_fiducials(self, draw: ImageDraw.ImageDraw) -> None:
        s = self._s
        margin_x = s(22)
        margin_top = s(22)
        margin_bottom = s(18)
        length = s(14)
        current_accent = self._hud_accent(self._last_state_name or "Lorenz", HUD_RED)
        color = self._rgba(current_accent, 160)
        for x, y, dx, dy in (
            (margin_x, margin_top, 1, 1),
            (self.width - margin_x, margin_top, -1, 1),
            (margin_x, self.height - margin_bottom, 1, -1),
            (self.width - margin_x, self.height - margin_bottom, -1, -1),
        ):
            draw.line((x, y, x + length * dx, y), fill=color, width=1)
            draw.line((x, y, x, y + length * dy), fill=color, width=1)

    def control_hit_test(self, attractor_names: Sequence[str], position: tuple[int, int]) -> Optional[str]:
        layout = self._hud_layout(attractor_names)
        x, y = position
        panel_left, panel_top, panel_right, panel_bottom = layout.params_panel
        if not (panel_left <= x <= panel_right and panel_top <= y <= panel_bottom):
            return None
        for slider_id, (left, top, right, bottom) in layout.slider_tracks.items():
            if left <= x <= right and top - 16 <= y <= bottom + 10:
                return slider_id
        return None

    def navigation_hit_test(self, attractor_names: Sequence[str], position: tuple[int, int]) -> Optional[int]:
        layout = self._hud_layout(attractor_names)
        for index, rect in enumerate(layout.nav_rows):
            if self._contains(rect, position):
                return index
        return None

    def shortcuts_toggle_hit_test(
        self,
        attractor_names: Sequence[str],
        position: tuple[int, int],
        show_shortcuts: bool,
    ) -> bool:
        layout = self._hud_layout(attractor_names)
        return self._contains(layout.shortcuts_toggle, position)

    def pip_hit_test(self, attractor_names: Sequence[str], position: tuple[int, int]) -> bool:
        layout = self._hud_layout(attractor_names)
        return self._contains(layout.pip_rect, position)

    def reset_button_hit_test(self, attractor_names: Sequence[str], position: tuple[int, int]) -> bool:
        layout = self._hud_layout(attractor_names)
        return self._contains(layout.reset_button, position)

    def control_value_for_position(self, attractor_names: Sequence[str], slider_id: str, x_position: int) -> float:
        layout = self._hud_layout(attractor_names)
        left, _top, right, _bottom = layout.slider_tracks[slider_id]
        minimum, maximum = dict(CONTROL_SLIDERS)[slider_id]
        ratio = 0.0 if right <= left else max(0.0, min(1.0, (x_position - left) / (right - left)))
        value = minimum + (maximum - minimum) * ratio
        return round(value, 2)

    def _wrap_lines(self, text: str, width: int) -> list[str]:
        lines: list[str] = []
        for paragraph in text.splitlines() or [text]:
            wrapped = textwrap.wrap(paragraph, width=width) or [paragraph]
            lines.extend(wrapped)
        return lines

    def _wrap_text_pixels(self, text: str, max_width: int, font) -> list[str]:
        lines: list[str] = []
        for paragraph in text.splitlines() or [text]:
            words = paragraph.split()
            if not words:
                lines.append("")
                continue

            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if self._text_width(candidate, font) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
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

    def _text_height(self, text: str, font) -> int:
        bbox = font.getbbox(text or " ")
        return max(1, bbox[3] - bbox[1] + 2)

    def _rgba(self, color: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
        return color[0], color[1], color[2], max(0, min(255, int(alpha)))

    def _draw_dot(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        *,
        glow: int = 0,
        alpha: int = 255,
    ) -> None:
        x, y = center
        if glow > 0:
            glow_fill = self._rgba(color, max(24, min(96, alpha // 3)))
            draw.ellipse((x - glow, y - glow, x + glow, y + glow), fill=glow_fill)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=self._rgba(color, alpha))

    def _contains(self, rect: tuple[int, int, int, int], position: tuple[int, int]) -> bool:
        left, top, right, bottom = rect
        x, y = position
        return left <= x <= right and top <= y <= bottom

    def _format_coordinate(self, value: float) -> str:
        return f"{value:>7.3f}".replace("-", "−")

    def _study_title_lines(self, title: str) -> list[str]:
        stripped = title.strip()
        if stripped.lower().endswith(" attractor"):
            return [stripped[:-10].upper(), "ATTRACTOR"]
        return self._wrap_lines(stripped.upper(), 12)[:2]

    def _format_equation_lines(self, equation: str, max_width: int) -> list[str]:
        formatted = equation
        for source, target in (
            ("xdot", "ẋ"),
            ("ydot", "ẏ"),
            ("zdot", "ż"),
            ("sigma", "σ"),
            ("rho", "ρ"),
            ("beta", "β"),
            ("lambda", "λ"),
            ("omega", "ω"),
            ("alpha", "α"),
            ("epsilon", "ε"),
            ("*", "·"),
        ):
            formatted = formatted.replace(source, target)

        lines: list[str] = []
        for line in formatted.splitlines():
            lines.extend(self._wrap_text_pixels(line, max_width, self._font_mono_9))
        return lines

    def _format_timestamp(self, seconds: float) -> str:
        whole = max(0, int(seconds))
        hours = whole // 3600
        minutes = (whole % 3600) // 60
        secs = whole % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _scanline_layer(self) -> Image.Image:
        key = (self.width, self.height)
        cached = self._scanlines_cache.get(key)
        if cached is None:
            rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            rgba[3::4, :, 3] = 15
            cached = Image.fromarray(rgba, mode="RGBA")
            self._scanlines_cache[key] = cached
        return cached

    def _vignette_layer(self) -> Image.Image:
        key = (self.width, self.height)
        cached = self._vignette_cache.get(key)
        if cached is None:
            y, x = np.ogrid[: self.height, : self.width]
            nx = (x - self.width * 0.5) / (self.width * 0.5)
            ny = (y - self.height * 0.5) / (self.height * 0.5)
            distance = np.sqrt(nx * nx + ny * ny)
            alpha = np.clip((distance - 0.74) / 0.26, 0.0, 1.0)
            alpha = np.power(alpha, 1.65)
            rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            rgba[:, :, 3] = np.clip(alpha * 142.0, 0.0, 255.0).astype(np.uint8)
            cached = Image.fromarray(rgba, mode="RGBA")
            self._vignette_cache[key] = cached
        return cached

    def _draw_camera_frame(
        self,
        frame_bgr: np.ndarray,
        left_landmarks: Optional[Sequence[tuple[float, float, float]]],
        right_landmarks: Optional[Sequence[tuple[float, float, float]]],
        left_caption: str,
        right_caption: str,
        *,
        rect: tuple[int, int, int, int] | None = None,
        opacity: float = 0.94,
        darken: float = 1.0,
    ) -> None:
        target_rect = rect or (
            self.width - self._s(28) - PIP_W,
            self.height - self._s(24) - PIP_H,
            self.width - self._s(28),
            self.height - self._s(24),
        )
        left, top, right, bottom = target_rect
        width = max(1, right - left)
        height = max(1, bottom - top)
        if self._cv2 is None:
            try:
                import cv2
            except ImportError:  # pragma: no cover - optional dependency
                cv2 = None
            self._cv2 = cv2
        if self._cv2 is not None:
            preview = self._cv2.resize(frame_bgr, (width, height))
            preview_rgb = self._cv2.cvtColor(preview, self._cv2.COLOR_BGR2RGB)
        else:  # pragma: no cover - only used when cv2 is unavailable
            preview = np.asarray(frame_bgr)
            preview_rgb = np.resize(preview[:, :, ::-1], (height, width, 3))
        if left_landmarks or right_landmarks:
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            if left_landmarks:
                draw_hand_skeleton(overlay, left_landmarks, width, height, caption=left_caption)
            if right_landmarks:
                draw_hand_skeleton(overlay, right_landmarks, width, height, caption=right_caption)
            overlay_rgba = pygame.image.tobytes(overlay, "RGBA")
            overlay_rgba = np.frombuffer(overlay_rgba, dtype=np.uint8).reshape(height, width, 4)
            alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
            preview_rgb = (
                preview_rgb.astype(np.float32) * (1.0 - alpha)
                + overlay_rgba[:, :, :3].astype(np.float32) * alpha
            ).astype(np.uint8)
        if darken < 1.0:
            preview_rgb = np.clip(preview_rgb.astype(np.float32) * darken, 0.0, 255.0).astype(np.uint8)
        preview_rgb = np.flipud(preview_rgb).astype(np.uint8, copy=False)
        self.camera_texture.write(preview_rgb.tobytes())
        self._render_quad_texture(self.camera_texture, (left, top, width, height), opacity=opacity)

    def tick(self) -> float:
        pygame.display.flip()
        self.clock.tick(FPS)
        return self.clock.get_fps()

    def quit(self) -> None:
        if self.gpu_stepper is not None:
            self.gpu_stepper.release()
        self.camera_texture.release()
        self.background_texture.release()
        self.fog_texture.release()
        self.overlay_texture.release()
        self.background_vao.release()
        self.quad_vao.release()
        self.quad_buffer.release()
        self.background_program.release()
        self.quad_program.release()
        self.point_vao.release()
        self.point_buffer.release()
        self.point_program.release()
        self.ctx.release()
        pygame.quit()
