from __future__ import annotations

import unittest

import numpy as np

from attractors.aizawa import AizawaAttractor
from attractors.chen import ChenAttractor
from attractors.dadras import DadrasAttractor
from attractors.langford import LangfordAttractor
from attractors.lorenz import LorenzAttractor
from attractors.manager import AttractorManager, create_active_attractor, normalize_points, perspective_project, rotation_matrix
from attractors.sprott_b import SprottBAttractor
from attractors.thomas import ThomasAttractor


class AttractorTests(unittest.TestCase):
    def test_each_active_attractor_steps_without_nan(self) -> None:
        attractors = [
            LorenzAttractor(),
            AizawaAttractor(),
            SprottBAttractor(),
            ThomasAttractor(),
            DadrasAttractor(),
            ChenAttractor(),
            LangfordAttractor(),
        ]
        for attractor in attractors:
            for _ in range(120):
                point = attractor.step(0.005)
                self.assertEqual(point.shape, (3,))
                self.assertTrue(np.isfinite(point).all(), attractor.name)

    def test_manager_exposes_seven_active_attractors(self) -> None:
        manager = AttractorManager()
        self.assertEqual(manager.total, 7)
        self.assertEqual(
            manager.names,
            ("Lorenz", "Aizawa", "Sprott B", "Thomas", "Dadras", "Chen", "Langford"),
        )

    def test_manager_projection_shapes_are_stable(self) -> None:
        manager = AttractorManager()
        manager.step_many(0.005, 64)
        points_2d, depths = manager.get_projected_trail(48, 25.0, -10.0, 5.0, 1.4, (1280, 720))
        self.assertEqual(points_2d.shape, (48, 2))
        self.assertEqual(depths.shape, (48,))
        self.assertTrue(np.isfinite(points_2d).all())
        self.assertTrue(np.isfinite(depths).all())

    def test_switching_resets_selected_attractor_trail(self) -> None:
        manager = AttractorManager()
        manager.step_many(0.005, 12)
        self.assertGreater(manager.count, 0)
        manager.switch_to(3)
        self.assertEqual(manager.name, "Thomas")
        self.assertEqual(manager.count, 0)
        manager.step_many(0.005, 5)
        manager.switch_to(3)
        self.assertEqual(manager.count, 0)

    def test_reset_all_clears_all_trails(self) -> None:
        manager = AttractorManager()
        for _ in range(manager.total):
            manager.step_many(0.005, 8)
            manager.switch_relative(1)
        manager.reset_all()
        self.assertTrue(all(manager.get_recent_trail(index=index).size == 0 for index in range(manager.total)))

    def test_render_data_and_normalization_helpers(self) -> None:
        manager = AttractorManager()
        manager.step_many(0.005, 96)
        render_points, ages = manager.get_render_data(64)
        self.assertEqual(render_points.shape, (64, 3))
        self.assertEqual(ages.shape, (64,))
        self.assertTrue(np.isfinite(render_points).all())
        self.assertTrue(np.isfinite(ages).all())

        points = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], dtype=np.float32)
        normalized = normalize_points(points)
        rotation = rotation_matrix(0.0, 0.0, 0.0)
        projected, depths = perspective_project(normalized, rotation, 1.0, (640, 480))
        self.assertEqual(projected.shape, (3, 2))
        self.assertEqual(depths.shape, (3,))

    def test_factory_creates_langford_case_insensitively(self) -> None:
        attractor = create_active_attractor("langford")
        self.assertEqual(attractor.name, "Langford")
        samples = attractor.sample_points(32, dt=0.005, burn_in=64)
        self.assertEqual(samples.shape, (32, 3))
        self.assertTrue(np.isfinite(samples).all())


if __name__ == "__main__":
    unittest.main()
