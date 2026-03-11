from __future__ import annotations

import unittest

import numpy as np

from attractors.aizawa import AizawaAttractor
from attractors.chen import ChenAttractor
from attractors.dadras import DadrasAttractor
from attractors.halvorsen import HalvorsenAttractor
from attractors.langford import LangfordAttractor
from attractors.lorenz import LorenzAttractor
from attractors.manager import AttractorManager, create_active_attractor, normalize_points, perspective_project, rotation_matrix
from attractors.rossler import RosslerAttractor
from attractors.sprott_b import SprottBAttractor
from attractors.thomas import ThomasAttractor


class AttractorTests(unittest.TestCase):
    def make_manager(self, capacity: int = 4096) -> AttractorManager:
        return AttractorManager(capacity=capacity)

    def test_each_active_attractor_steps_without_nan(self) -> None:
        attractors = [
            LorenzAttractor(),
            AizawaAttractor(),
            SprottBAttractor(),
            ThomasAttractor(),
            DadrasAttractor(),
            ChenAttractor(),
            LangfordAttractor(),
            RosslerAttractor(),
            HalvorsenAttractor(),
        ]
        for attractor in attractors:
            for _ in range(120):
                point = attractor.step(0.005)
                self.assertEqual(point.shape, (3,))
                self.assertTrue(np.isfinite(point).all(), attractor.name)

    def test_manager_exposes_all_active_attractors(self) -> None:
        manager = self.make_manager()
        self.assertEqual(manager.total, 9)
        self.assertEqual(
            manager.names,
            ("Lorenz", "Aizawa", "Sprott B", "Thomas", "Dadras", "Chen", "Langford", "Rossler", "Halvorsen"),
        )

    def test_manager_uses_hud_accent_palette(self) -> None:
        manager = self.make_manager()
        expected_colors = {
            "Lorenz": (230, 57, 70),
            "Aizawa": (255, 209, 102),
            "Sprott B": (6, 214, 160),
            "Thomas": (255, 107, 157),
            "Dadras": (76, 201, 240),
            "Chen": (199, 125, 255),
            "Langford": (17, 138, 178),
            "Rossler": (244, 162, 97),
            "Halvorsen": (82, 183, 136),
        }

        for index, name in enumerate(manager.names):
            manager.switch_to(index)
            self.assertEqual(manager.color, expected_colors[name])

    def test_active_placards_use_attractor_specific_medium_descriptions(self) -> None:
        manager = self.make_manager()
        expected_media = {
            "Lorenz": "Atmospheric convection model,\ndissipative chaotic flow",
            "Aizawa": "Autonomous nonlinear flow,\ntoroidal strange attractor",
            "Sprott B": "Minimal quadratic flow,\nSprott class-B chaos",
            "Thomas": "Cyclically symmetric flow,\nthree coupled sine states",
            "Dadras": "Polynomial chaotic flow,\nstate-multiplying feedback",
            "Chen": "Lorenz-family chaotic flow,\nquadratic dissipative system",
            "Langford": "Torus-breakdown oscillator,\nfolded nonlinear flow",
            "Rossler": "Single-scroll spiral flow,\ncontinuous-time oscillator",
            "Halvorsen": "Symmetric quadratic flow,\nthree-lobed chaotic system",
        }

        for index, name in enumerate(manager.names):
            manager.switch_to(index)
            self.assertEqual(manager.placard.medium, expected_media[name])

    def test_active_placards_include_attractor_equations(self) -> None:
        manager = self.make_manager()
        expected_equations = {
            "Lorenz": "xdot = sigma(y - x)\nydot = x(rho - z) - y\nzdot = x*y - beta*z",
            "Aizawa": "xdot = (z - b)*x - d*y\nydot = d*x + (z - b)*y\nzdot = c + a*z - z^3/3 -\n(x^2 + y^2)(1 + e*z) + f*z*x^3",
            "Sprott B": "xdot = a*y*z\nydot = x - b*y\nzdot = 1 - x*y",
            "Thomas": "xdot = sin(y) - b*x\nydot = sin(z) - b*y\nzdot = sin(x) - b*z",
            "Dadras": "xdot = y - p*x + q*y*z\nydot = r*y - x*z + z\nzdot = c*x*y - e*z",
            "Chen": "xdot = a(y - x)\nydot = (c - a)*x - x*z + c*y\nzdot = x*y - b*z",
            "Langford": "xdot = (z - beta)*x - omega*y\nydot = omega*x + (z - beta)*y\nzdot = lambda + alpha*z - z^3/3 -\n(x^2 + y^2)(1 + rho*z) + epsilon*z*x^3",
            "Rossler": "xdot = -(y + z)\nydot = x + a*y\nzdot = b + z(x - c)",
            "Halvorsen": "xdot = -a*x - 4*y - 4*z - y^2\nydot = -a*y - 4*z - 4*x - z^2\nzdot = -a*z - 4*x - 4*y - x^2",
        }

        for index, name in enumerate(manager.names):
            manager.switch_to(index)
            self.assertEqual(manager.placard.equation, expected_equations[name])

    def test_manager_projection_shapes_are_stable(self) -> None:
        manager = self.make_manager()
        manager.step_many(0.005, 64)
        points_2d, depths = manager.get_projected_trail(48, 25.0, -10.0, 5.0, 1.4, (1280, 720))
        self.assertEqual(points_2d.shape, (48, 2))
        self.assertEqual(depths.shape, (48,))
        self.assertTrue(np.isfinite(points_2d).all())
        self.assertTrue(np.isfinite(depths).all())

    def test_switching_resets_selected_attractor_trail(self) -> None:
        manager = self.make_manager()
        manager.step_many(0.005, 12)
        self.assertGreater(manager.count, 0)
        manager.switch_to(3)
        self.assertEqual(manager.name, "Thomas")
        self.assertEqual(manager.count, 0)
        manager.step_many(0.005, 5)
        manager.switch_to(3)
        self.assertEqual(manager.count, 0)

    def test_reset_all_clears_all_trails(self) -> None:
        manager = self.make_manager()
        for _ in range(manager.total):
            manager.step_many(0.005, 8)
            manager.switch_relative(1)
        manager.reset_all()
        self.assertTrue(all(manager.get_recent_trail(index=index).size == 0 for index in range(manager.total)))

    def test_render_data_and_normalization_helpers(self) -> None:
        manager = self.make_manager()
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

    def test_dadras_default_seed_enters_full_3d_motion(self) -> None:
        attractor = DadrasAttractor()
        samples = attractor.fill_samples(0.005, 512)

        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(float(np.std(samples[:, 1])), 1e-3)
        self.assertGreater(float(np.std(samples[:, 2])), 1e-3)

    def test_prime_current_trail_fills_large_buffer_and_advances_state(self) -> None:
        manager = self.make_manager(capacity=1024)
        initial_state = manager.state_vector
        manager.prime_current_trail(dt=0.005, sample_count=512, burn_in=128)
        self.assertEqual(manager.count, 512)
        self.assertEqual(manager.get_recent_trail().shape, (512, 3))
        self.assertNotEqual(manager.state_vector, initial_state)
        render_points, ages = manager.get_render_data(64)
        self.assertEqual(render_points.shape, (64, 3))
        self.assertEqual(ages.shape, (64,))

    def test_sample_points_rejects_invalid_sampling_parameters(self) -> None:
        attractor = LorenzAttractor()
        with self.assertRaisesRegex(ValueError, "burn_in"):
            attractor.sample_points(16, dt=0.005, burn_in=-1)
        with self.assertRaisesRegex(ValueError, "sample_stride"):
            attractor.sample_points(16, dt=0.005, sample_stride=0)


if __name__ == "__main__":
    unittest.main()
