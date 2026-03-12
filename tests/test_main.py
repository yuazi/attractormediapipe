from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


class MainModuleTests(unittest.TestCase):
    def test_module_import_does_not_parse_process_argv(self) -> None:
        sys.modules.pop("main", None)
        with mock.patch.object(sys, "argv", ["prog", "--unexpected"]):
            module = importlib.import_module("main")

        self.assertTrue(callable(module.main))
        self.assertTrue(module.parse_args(["--demo"]).demo)

        sys.modules.pop("main", None)

    def test_points_per_second_hits_configured_rate_at_max_speed(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        self.assertAlmostEqual(
            module._points_per_second_for_speed(module.SPEED_RANGE[1]) * 60.0,
            module.MAX_SPEED_POINTS_PER_MINUTE,
        )
        self.assertLess(module._points_per_second_for_speed(module.SPEED_RANGE[0]), module._points_per_second_for_speed(1.0))

        sys.modules.pop("main", None)

    def test_sample_budget_reaches_configured_points_per_minute_at_max_speed(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        sample_budget = 0.0
        total_steps = 0
        for _ in range(60 * 60):
            steps, sample_budget = module._consume_sample_budget(sample_budget, module.SPEED_RANGE[1], 1.0 / 60.0)
            total_steps += steps

        self.assertEqual(total_steps, module.MAX_SPEED_POINTS_PER_MINUTE)
        self.assertLess(sample_budget, 1.0)

        sys.modules.pop("main", None)

    def test_adjust_fog_clamps_to_range(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        controls = module.ControlState()
        module._adjust_fog(controls, 10.0)
        controls.smooth(alpha=1.0)
        self.assertEqual(controls.fog, module.FOG_RANGE[1])

        module._adjust_fog(controls, -10.0)
        controls.smooth(alpha=1.0)
        self.assertEqual(controls.fog, module.FOG_RANGE[0])

        sys.modules.pop("main", None)

    def test_slider_control_updates_fog_target(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        controls = module.ControlState()
        module._apply_slider_control(controls, "fog", 0.35)
        controls.smooth(alpha=1.0)

        self.assertAlmostEqual(controls.fog, 0.35)

        sys.modules.pop("main", None)

    def test_live_render_data_uses_fixed_trail_length(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        class FakeManager:
            def __init__(self) -> None:
                self.limit = None

            def get_render_data(self, limit: int):
                self.limit = limit
                return "positions", "ages"

        manager = FakeManager()
        positions, ages = module._get_live_render_data(manager)

        self.assertEqual((positions, ages), ("positions", "ages"))
        self.assertEqual(manager.limit, module.FIXED_TRAIL_LENGTH)

        sys.modules.pop("main", None)

    def test_run_snapshot_export_forwards_snapshot_dimensions(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        captured = {}

        def fake_export(request):
            captured["request"] = request
            return "/tmp/wallpaper.png"

        args = module.parse_args(
            [
                "--snapshot-only",
                "--attractor",
                "Lorenz",
                "--snapshot-width",
                "7680",
                "--snapshot-height",
                "4320",
            ]
        )

        with mock.patch("renderer.export_attractor_snapshot", side_effect=fake_export):
            with mock.patch("builtins.print"):
                result = module.run_snapshot_export(args)

        self.assertEqual(result, "/tmp/wallpaper.png")
        self.assertEqual(captured["request"].width, 7680)
        self.assertEqual(captured["request"].height, 4320)

        sys.modules.pop("main", None)

    def test_maybe_create_camera_session_disables_camera_when_tracker_init_fails(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        class FakeCapture:
            def __init__(self) -> None:
                self.released = False

            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                self.released = True

            def set(self, *_args) -> bool:
                return True

        fake_capture = FakeCapture()
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda _index: fake_capture,
            CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_FRAME_HEIGHT=2,
            CAP_PROP_FPS=3,
            CAP_PROP_BUFFERSIZE=4,
        )

        import hands.tracker as tracker_module

        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            with mock.patch.object(tracker_module, "MEDIAPIPE_AVAILABLE", True):
                with mock.patch.object(tracker_module, "HandTracker", side_effect=RuntimeError("missing model")):
                    with mock.patch("builtins.print") as print_mock:
                        session = module.maybe_create_camera_session(module.parse_args([]))

        self.assertIsNone(session)
        self.assertTrue(fake_capture.released)
        print_mock.assert_called_once()

        sys.modules.pop("main", None)


if __name__ == "__main__":
    unittest.main()
