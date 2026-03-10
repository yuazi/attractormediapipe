from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pygame
from PIL import Image

from config import (
    BACKGROUND_COLOR,
    CAPTION,
    FPS,
    PIP_BORDER_COLOR,
    PIP_H,
    PIP_MARGIN,
    PIP_W,
    SCREENSHOT_PREFIX,
    WIN_H,
    WIN_W,
)
from hands.skeleton import draw_hand_skeleton

from .bloom import BloomPass
from .hud import HUDRenderer
from .particles import ParticleRenderer


@dataclass
class SceneState:
    points_2d: np.ndarray
    depths: np.ndarray
    attractor_name: str
    attractor_color: Tuple[int, int, int]
    placard_title: str
    placard_year: str
    placard_medium: str
    placard_params: Sequence[Tuple[str, str]]
    luminosity: float
    speed: float
    scale: float
    trail_len: int
    particle_count: int
    attractor_names: Sequence[str]
    attractor_index: int
    attractor_state: Tuple[float, float, float]
    point_count: int
    fps: float
    left_detected: bool
    right_detected: bool
    bloom_enabled: bool
    show_overlay: bool
    show_camera: bool
    focus_mode: bool
    active_slider: Optional[str]
    particle_input_active: bool
    particle_input_text: str
    pip_frame: Optional[np.ndarray]
    left_landmarks: Optional[Sequence[Tuple[float, float, float]]]
    right_landmarks: Optional[Sequence[Tuple[float, float, float]]]


class SceneRenderer:
    def __init__(self, width: int = WIN_W, height: int = WIN_H) -> None:
        pygame.init()
        pygame.display.set_caption(CAPTION)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.particle_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.particles = ParticleRenderer()
        self.bloom = BloomPass(width, height)
        self.hud = HUDRenderer(width, height)
        self._pip_surface = pygame.Surface((PIP_W, PIP_H)).convert()
        self._pip_surface.set_alpha(225)
        self._pip_overlay = pygame.Surface((PIP_W, PIP_H), pygame.SRCALPHA)
        self._pip_border = pygame.Surface((PIP_W, PIP_H), pygame.SRCALPHA)
        self._last_pip_frame_id: int | None = None
        pygame.draw.rect(self._pip_border, PIP_BORDER_COLOR, self._pip_border.get_rect(), 2, border_radius=10)
        try:
            import cv2
        except ImportError:  # pragma: no cover - optional dependency
            cv2 = None
        self._cv2 = cv2

    def draw(self, state: SceneState) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        self.particle_surface.fill((0, 0, 0, 0))
        self.particles.draw(
            self.particle_surface,
            state.points_2d,
            state.depths,
            state.attractor_color,
            state.luminosity,
            state.particle_count,
        )
        self.screen.blit(self.particle_surface, (0, 0))
        if state.bloom_enabled:
            self.bloom.apply(self.screen, self.particle_surface)

        if (state.show_camera or state.focus_mode) and state.pip_frame is not None:
            self._draw_pip(
                state.pip_frame,
                state.left_landmarks,
                state.right_landmarks,
                left_caption="Speed",
                right_caption="Scale",
            )

        self.hud.draw(
            self.screen,
            attractor_name=state.attractor_name,
            attractor_color=state.attractor_color,
            placard_title=state.placard_title,
            placard_year=state.placard_year,
            placard_medium=state.placard_medium,
            placard_params=state.placard_params,
            attractor_names=state.attractor_names,
            attractor_index=state.attractor_index,
            attractor_state=state.attractor_state,
            point_count=state.point_count,
            speed=state.speed,
            scale=state.scale,
            luminosity=state.luminosity,
            trail_len=state.trail_len,
            particle_count=state.particle_count,
            fps=state.fps,
            left_detected=state.left_detected,
            right_detected=state.right_detected,
            show_overlay=state.show_overlay,
            focus_mode=state.focus_mode,
            active_slider=state.active_slider,
            particle_input_active=state.particle_input_active,
            particle_input_text=state.particle_input_text,
        )

    def _draw_pip(
        self,
        frame_bgr: np.ndarray,
        left_landmarks: Optional[Sequence[Tuple[float, float, float]]],
        right_landmarks: Optional[Sequence[Tuple[float, float, float]]],
        *,
        left_caption: str,
        right_caption: str,
    ) -> None:
        frame_id = id(frame_bgr)
        if frame_id != self._last_pip_frame_id:
            if self._cv2 is not None:
                preview = self._cv2.resize(frame_bgr, (PIP_W, PIP_H))
                preview_rgb = self._cv2.cvtColor(preview, self._cv2.COLOR_BGR2RGB)
            else:  # pragma: no cover - only used when cv2 is unavailable
                preview = np.asarray(frame_bgr)
                preview_rgb = preview[:, :, ::-1]
                preview_rgb = np.resize(preview_rgb, (PIP_H, PIP_W, 3))
            pygame.surfarray.blit_array(self._pip_surface, np.transpose(preview_rgb, (1, 0, 2)))
            self._last_pip_frame_id = frame_id

        x = self.width - PIP_W - (PIP_MARGIN + 24)
        y = self.height - PIP_H - (PIP_MARGIN + 36)
        self.screen.blit(self._pip_surface, (x, y))

        self._pip_overlay.fill((0, 0, 0, 0))
        if left_landmarks:
            draw_hand_skeleton(self._pip_overlay, left_landmarks, PIP_W, PIP_H, caption=left_caption)
        if right_landmarks:
            draw_hand_skeleton(self._pip_overlay, right_landmarks, PIP_W, PIP_H, caption=right_caption)
        self.screen.blit(self._pip_overlay, (x, y))
        self.screen.blit(self._pip_border, (x, y))

    def tick(self) -> float:
        pygame.display.flip()
        self.clock.tick(FPS)
        return self.clock.get_fps()

    def save_screenshot(self, path: str = "") -> str:
        filename = path or f"{SCREENSHOT_PREFIX}_{int(time.time())}.png"
        rgb = pygame.surfarray.array3d(self.screen)
        rgb = np.transpose(rgb, (1, 0, 2))
        Image.fromarray(rgb).save(filename)
        return filename

    def quit(self) -> None:
        pygame.quit()
