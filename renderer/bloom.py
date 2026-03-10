from __future__ import annotations

import numpy as np
import pygame

from config import BLOOM_ALPHA, BLOOM_BLUR_PASSES, BLOOM_DOWNSAMPLE


class BloomPass:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.downsample = BLOOM_DOWNSAMPLE
        self.lowres_width = max(1, width // self.downsample)
        self.lowres_height = max(1, height // self.downsample)
        self._downsample_surface = pygame.Surface((self.lowres_width, self.lowres_height)).convert()
        self._bloom_surface = pygame.Surface((self.lowres_width, self.lowres_height)).convert()
        self._upsampled_surface = pygame.Surface((self.width, self.height)).convert()
        self._upsampled_surface.set_alpha(BLOOM_ALPHA)

    def apply(self, target_surface: pygame.Surface, particle_surface: pygame.Surface) -> None:
        pygame.transform.smoothscale(
            particle_surface,
            (self.lowres_width, self.lowres_height),
            self._downsample_surface,
        )
        rgb = pygame.surfarray.array3d(self._downsample_surface).astype(np.float32)
        rgb = np.transpose(rgb, (1, 0, 2))
        blurred = rgb
        for _ in range(BLOOM_BLUR_PASSES):
            blurred = self._box_blur(blurred)
        pygame.surfarray.blit_array(
            self._bloom_surface,
            np.transpose(np.clip(blurred, 0, 255).astype(np.uint8), (1, 0, 2)),
        )
        pygame.transform.smoothscale(
            self._bloom_surface,
            (self.width, self.height),
            self._upsampled_surface,
        )
        target_surface.blit(self._upsampled_surface, (0, 0), special_flags=pygame.BLEND_ADD)

    def _box_blur(self, rgb: np.ndarray) -> np.ndarray:
        padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
        return (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
