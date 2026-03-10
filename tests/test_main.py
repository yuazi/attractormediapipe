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


if __name__ == "__main__":
    unittest.main()
