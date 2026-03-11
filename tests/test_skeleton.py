from __future__ import annotations

import unittest

import pygame

from hands.skeleton import draw_hand_skeleton


def make_hand() -> list[tuple[float, float, float]]:
    points = [(0.5, 0.5, 0.0) for _ in range(21)]
    points[4] = (0.32, 0.56, 0.0)
    points[8] = (0.54, 0.56, 0.0)
    points[16] = (0.46, 0.34, 0.0)
    points[17] = (0.64, 0.52, 0.0)
    points[18] = (0.72, 0.44, 0.0)
    points[19] = (0.78, 0.34, 0.0)
    points[20] = (0.82, 0.26, 0.0)
    return points


class SkeletonTests(unittest.TestCase):
    def test_draw_hand_skeleton_renders_ring_line_and_only_pinky_tip_marker(self) -> None:
        surface = pygame.Surface((320, 180), pygame.SRCALPHA)
        draw_hand_skeleton(surface, make_hand(), 320, 180, caption=None)

        pinky_tip = surface.get_at((262, 46))
        pinky_joint = surface.get_at((249, 61))
        ring_tip = surface.get_at((147, 61))
        ring_line_midpoint = surface.get_at((125, 81))

        self.assertGreater(pinky_tip.a, 0)
        self.assertEqual(pinky_joint.a, 0)
        self.assertGreater(ring_tip.a, 0)
        self.assertGreater(ring_line_midpoint.a, 0)


if __name__ == "__main__":
    unittest.main()
