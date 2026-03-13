from __future__ import annotations

import importlib
import sys
import threading
import time
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

    def test_renderer_snapshot_exports_do_not_eagerly_import_scene(self) -> None:
        module_names = ("renderer", "renderer.scene", "renderer.snapshot")
        original_modules = {name: sys.modules.get(name) for name in module_names}
        try:
            for name in module_names:
                sys.modules.pop(name, None)

            renderer = importlib.import_module("renderer")

            self.assertNotIn("renderer.scene", sys.modules)
            self.assertTrue(callable(renderer.export_attractor_snapshot))
            self.assertNotIn("renderer.scene", sys.modules)
            self.assertTrue(callable(renderer.SnapshotRequest))
            self.assertNotIn("renderer.scene", sys.modules)
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
            for name, module in original_modules.items():
                if module is not None:
                    sys.modules[name] = module

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

    def test_adjust_yaw_and_pitch_clamp_to_range(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        controls = module.ControlState()
        module._adjust_yaw(controls, 10_000.0)
        module._adjust_pitch(controls, -10_000.0)
        controls.smooth(alpha=1.0)

        self.assertEqual(controls.yaw, module.YAW_RANGE[1])
        self.assertEqual(controls.pitch, module.PITCH_RANGE[0])

        module._adjust_yaw(controls, -10_000.0)
        module._adjust_pitch(controls, 10_000.0)
        controls.smooth(alpha=1.0)

        self.assertEqual(controls.yaw, module.YAW_RANGE[0])
        self.assertEqual(controls.pitch, module.PITCH_RANGE[1])

        sys.modules.pop("main", None)

    def test_apply_held_rotation_controls_accumulates_over_frame_time(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        controls = module.ControlState()
        module._apply_held_rotation_controls(controls, horizontal=1.0, vertical=-1.0, frame_delta=0.5)
        controls.smooth(alpha=1.0)

        self.assertEqual(controls.yaw, module.KEYBOARD_YAW_HOLD_RATE * 0.5)
        self.assertEqual(controls.pitch, -module.KEYBOARD_PITCH_HOLD_RATE * 0.5)

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
            return types.SimpleNamespace(clean_path="/tmp/wallpaper_clean.png", textured_path="/tmp/wallpaper_textured.png")

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

        self.assertEqual(result.clean_path, "/tmp/wallpaper_clean.png")
        self.assertEqual(result.textured_path, "/tmp/wallpaper_textured.png")
        self.assertEqual(captured["request"].width, 7680)
        self.assertEqual(captured["request"].height, 4320)

        sys.modules.pop("main", None)

    def test_open_camera_capture_auto_scan_falls_back_to_next_working_device(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        class FakeCapture:
            def __init__(self, *, opened: bool, readable: bool) -> None:
                self._opened = opened
                self._readable = readable
                self.released = False
                self.settings: list[tuple[int, int]] = []

            def isOpened(self) -> bool:
                return self._opened

            def read(self):
                return self._readable, object() if self._readable else None

            def release(self) -> None:
                self.released = True

            def set(self, prop: int, value: int) -> bool:
                self.settings.append((prop, value))
                return True

        captures = {
            0: FakeCapture(opened=False, readable=False),
            1: FakeCapture(opened=True, readable=False),
            2: FakeCapture(opened=True, readable=True),
        }
        open_order: list[int] = []

        def video_capture(index: int):
            open_order.append(index)
            return captures[index]

        fake_cv2 = types.SimpleNamespace(
            VideoCapture=video_capture,
            CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_FRAME_HEIGHT=2,
            CAP_PROP_FPS=3,
            CAP_PROP_BUFFERSIZE=4,
        )

        capture, index = module._open_camera_capture(fake_cv2, -1)

        self.assertIs(capture, captures[2])
        self.assertEqual(index, 2)
        self.assertEqual(open_order, [0, 1, 2])
        self.assertTrue(captures[0].released)
        self.assertTrue(captures[1].released)
        self.assertFalse(captures[2].released)
        self.assertIn((fake_cv2.CAP_PROP_FRAME_WIDTH, module.CAMERA_FRAME_WIDTH), captures[2].settings)
        self.assertIn((fake_cv2.CAP_PROP_FRAME_HEIGHT, module.CAMERA_FRAME_HEIGHT), captures[2].settings)
        self.assertIn((fake_cv2.CAP_PROP_FPS, module.CAMERA_CAPTURE_FPS), captures[2].settings)
        self.assertIn((fake_cv2.CAP_PROP_BUFFERSIZE, 1), captures[2].settings)

        sys.modules.pop("main", None)

    def test_camera_session_close_waits_for_active_read_before_releasing_capture(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        class FakeCapture:
            def __init__(self) -> None:
                self.entered_read = threading.Event()
                self.allow_return = threading.Event()
                self.released = False

            def read(self):
                self.entered_read.set()
                self.allow_return.wait(timeout=1.0)
                return False, None

            def release(self) -> None:
                self.released = True

        class FakeTracker:
            def __init__(self) -> None:
                self.closed = False

            def process(self, _frame):
                return {"left": None, "right": None}

            def close(self) -> None:
                self.closed = True

        capture = FakeCapture()
        tracker = FakeTracker()
        fake_cv2 = types.SimpleNamespace(flip=lambda frame, _axis: frame)
        session = module.CameraTrackerSession(capture, tracker, fake_cv2)

        self.assertTrue(capture.entered_read.wait(timeout=1.0))

        closer = threading.Thread(target=session.close)
        closer.start()
        time.sleep(0.05)
        self.assertFalse(capture.released)

        capture.allow_return.set()
        closer.join(timeout=1.0)

        self.assertFalse(closer.is_alive())
        self.assertTrue(capture.released)
        self.assertTrue(tracker.closed)

        sys.modules.pop("main", None)

    def test_camera_session_clears_snapshot_when_tracker_processing_fails(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        class FakeCapture:
            def __init__(self) -> None:
                self.released = False

            def read(self):
                return True, object()

            def release(self) -> None:
                self.released = True

        class FakeTracker:
            def __init__(self) -> None:
                self.closed = False
                self.process_called = threading.Event()

            def process(self, _frame):
                self.process_called.set()
                raise RuntimeError("tracker failed")

            def close(self) -> None:
                self.closed = True

        capture = FakeCapture()
        tracker = FakeTracker()
        fake_cv2 = types.SimpleNamespace(flip=lambda frame, _axis: frame)

        with mock.patch("builtins.print") as print_mock:
            session = module.CameraTrackerSession(capture, tracker, fake_cv2)
            self.assertTrue(tracker.process_called.wait(timeout=1.0))
            session._worker.join(timeout=1.0)

            self.assertFalse(session._worker.is_alive())
            self.assertFalse(session._running)
            snapshot = session.snapshot()
            self.assertIsNone(snapshot.frame)
            self.assertEqual(snapshot.hand_data, {"left": None, "right": None})
            print_mock.assert_called_once()
            self.assertIn("tracker failed", print_mock.call_args[0][0])

            session.close()

        self.assertTrue(capture.released)
        self.assertTrue(tracker.closed)

        sys.modules.pop("main", None)

    def test_maybe_create_camera_session_disables_camera_when_tracker_init_fails(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        class FakeCapture:
            def __init__(self) -> None:
                self.released = False

            def isOpened(self) -> bool:
                return True

            def read(self):
                return True, object()

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
