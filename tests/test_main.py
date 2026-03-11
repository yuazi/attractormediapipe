from __future__ import annotations

import importlib
import sys
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


if __name__ == "__main__":
    unittest.main()
