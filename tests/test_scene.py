from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from renderer.scene import SceneRenderer, SceneState


class DummyTexture:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, pixels: bytes) -> None:
        self.writes.append(pixels)


class DummyUniform:
    def __init__(self) -> None:
        self.value = None
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)


class DummyProgram(dict):
    def __getitem__(self, key: str) -> DummyUniform:
        if key not in self:
            self[key] = DummyUniform()
        return dict.__getitem__(self, key)


class DummyBuffer:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)


class DummyVertexArray:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render(self, **kwargs) -> None:
        self.calls.append(kwargs)


class DummyContext:
    def __init__(self) -> None:
        self.clear_calls: list[tuple[float, float, float, float]] = []
        self.blend_func = None

    def clear(self, red: float, green: float, blue: float, alpha: float) -> None:
        self.clear_calls.append((red, green, blue, alpha))


class SceneRendererTests(unittest.TestCase):
    def test_render_atmosphere_uses_fog_opacity_without_rebuilding_cache(self) -> None:
        overlay_texture = DummyTexture()
        probe = SimpleNamespace(
            width=8,
            height=6,
            _atmosphere_cache={},
            overlay_texture=overlay_texture,
            quad_calls=[],
        )
        probe._render_quad_texture = lambda texture, rect, opacity=1.0: probe.quad_calls.append((texture, rect, opacity))

        SceneRenderer._render_atmosphere(probe, (10, 20, 30), 0.25)
        cache_key = (probe.width, probe.height, (10, 20, 30))
        cached_pixels = probe._atmosphere_cache[cache_key]

        SceneRenderer._render_atmosphere(probe, (10, 20, 30), 0.75)

        self.assertIs(probe._atmosphere_cache[cache_key], cached_pixels)
        self.assertEqual(probe.quad_calls[0], (overlay_texture, (0, 0, 8, 6), 0.25))
        self.assertEqual(probe.quad_calls[1], (overlay_texture, (0, 0, 8, 6), 0.75))
        self.assertEqual(overlay_texture.writes[0], overlay_texture.writes[1])

    def test_draw_passes_fog_to_atmosphere_and_luminosity_to_point_shader(self) -> None:
        point_program = DummyProgram()
        point_buffer = DummyBuffer()
        point_vao = DummyVertexArray()
        ctx = DummyContext()
        layout = SimpleNamespace(pip_rect=(0, 0, 10, 10))
        atmosphere_calls: list[tuple[tuple[int, int, int], float]] = []

        renderer = SimpleNamespace(
            width=1280,
            height=720,
            ctx=ctx,
            moderngl=SimpleNamespace(ONE=1, POINTS=2),
            _fps_history=[30.0] * 5,
            point_program=point_program,
            point_buffer=point_buffer,
            point_vao=point_vao,
        )
        renderer._sync_window_size = lambda: None
        renderer._hud_accent = lambda _name, fallback: fallback
        renderer._hud_layout = lambda _names: layout
        renderer._render_atmosphere = lambda accent, fog: atmosphere_calls.append((accent, fog))
        renderer._ensure_point_capacity = lambda _byte_count: None
        renderer._draw_overlay = lambda *_args, **_kwargs: None
        renderer._draw_camera_frame = lambda *_args, **_kwargs: None

        state = SceneState(
            positions=np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
            ages=np.array([0.6], dtype=np.float32),
            attractor_name="Lorenz",
            attractor_color=(255, 88, 88),
            attractor_index=0,
            attractor_total=1,
            attractor_names=("Lorenz",),
            attractor_state=(1.0, 2.0, 3.0),
            placard_title="Lorenz Attractor",
            placard_year="E. N. Lorenz, 1963",
            placard_medium="Atmospheric convection model",
            placard_equation="xdot = y - x",
            placard_params=(("sigma", "10.0"),),
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            zoom=1.6,
            speed=1.0,
            fog=0.42,
            luminosity=0.73,
            point_count=1,
            fps=60.0,
            paused=False,
            exporting=False,
            export_message="",
            show_overlay=False,
            show_shortcuts=False,
            show_camera=False,
            focus_mode=False,
            active_slider=None,
            hover_slider=None,
            hover_nav_index=None,
            hover_reset=False,
            hover_shortcuts_toggle=False,
            hover_pip=False,
            left_detected=False,
            right_detected=False,
            pip_frame=None,
            left_landmarks=None,
            right_landmarks=None,
            left_pip_caption="Speed",
            right_pip_caption="Scale",
            time_value=1.5,
        )

        SceneRenderer.draw(renderer, state)

        self.assertEqual(atmosphere_calls, [((255, 88, 88), 0.42)])
        self.assertEqual(point_program["u_luminosity"].value, 0.73)
        self.assertEqual(point_vao.calls[0]["vertices"], 1)


if __name__ == "__main__":
    unittest.main()
