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

    def test_steps_for_speed_scales_with_speed(self) -> None:
        sys.modules.pop("main", None)
        module = importlib.import_module("main")

        self.assertEqual(module._steps_for_speed(0.1), 1)
        self.assertEqual(module._steps_for_speed(1.0), module.STEPS_PER_FRAME)
        self.assertGreater(module._steps_for_speed(3.5), module.STEPS_PER_FRAME)

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
