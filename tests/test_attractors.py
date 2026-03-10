from __future__ import annotations

import unittest

import numpy as np

from attractors.aizawa import AizawaAttractor
from attractors.chen import ChenAttractor
from attractors.dadras import DadrasAttractor
from attractors.halvorsen import HalvorsenAttractor
from attractors.lorenz import LorenzAttractor
from attractors.manager import AttractorManager, normalize_points, perspective_project, rotation_matrix
from attractors.rossler import RosslerAttractor
from attractors.sprott_b import SprottBAttractor
from attractors.thomas import ThomasAttractor


class AttractorTests(unittest.TestCase):
    def test_each_attractor_steps_without_nan(self) -> None:
        attractors = [
            LorenzAttractor(),
            RosslerAttractor(),
            HalvorsenAttractor(),
            ThomasAttractor(),
            DadrasAttractor(),
            AizawaAttractor(),
            SprottBAttractor(),
            ChenAttractor(),
        ]
        for attractor in attractors:
            for _ in range(120):
                point = attractor.step(0.005)
                self.assertEqual(point.shape, (3,))
                self.assertTrue(np.isfinite(point).all(), attractor.name)

    def test_manager_exposes_chen_in_navigation(self) -> None:
        manager = AttractorManager()
        self.assertEqual(manager.total, 8)
        self.assertEqual(manager.names[-1], "Chen")

    def test_manager_projection_shapes_are_stable(self) -> None:
        manager = AttractorManager()
        manager.step_many(0.005, 64)
        points_2d, depths = manager.get_projected_trail(48, 25.0, -10.0, 5.0, 1.4, (1280, 720))
        self.assertEqual(points_2d.shape, (48, 2))
        self.assertEqual(depths.shape, (48,))
        self.assertTrue(np.isfinite(points_2d).all())
        self.assertTrue(np.isfinite(depths).all())

    def test_manager_switch_and_reset_clear_trail(self) -> None:
        manager = AttractorManager()
        manager.step_many(0.005, 12)
        self.assertGreater(manager.count, 0)
        manager.switch_to(3)
        self.assertEqual(manager.name, "Thomas")
        self.assertEqual(manager.count, 0)
        manager.step_many(0.005, 5)
        manager.reset()
        self.assertEqual(manager.count, 0)

    def test_normalization_and_projection_helpers(self) -> None:
        points = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float64)
        normalized = normalize_points(points)
        rotation = rotation_matrix(0.0, 0.0, 0.0)
        projected, depths = perspective_project(normalized, rotation, 1.0, (640, 480))
        self.assertEqual(projected.shape, (3, 2))
        self.assertEqual(depths.shape, (3,))
        self.assertTrue(np.isfinite(projected).all())
        self.assertTrue(np.isfinite(depths).all())


if __name__ == "__main__":
    unittest.main()
