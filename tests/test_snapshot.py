from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from renderer.snapshot import SnapshotController, SnapshotRequest, ensure_snapshot_environment, export_attractor_snapshot, snapshot_filename


class SnapshotTests(unittest.TestCase):
    def test_snapshot_export_writes_4k_png(self) -> None:
        ensure_snapshot_environment()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "snapshot.png"
            request = SnapshotRequest(
                attractor_name="Lorenz",
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                zoom=1.6,
                time_value=0.0,
                luminosity=0.8,
                output_path=str(output_path),
                sample_count=3000,
                burn_in=500,
                sample_stride=1,
            )
            result = export_attractor_snapshot(request)
            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertGreater(os.path.getsize(output_path), 0)
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (3840, 2160))

    def test_snapshot_request_rejects_invalid_sampling_parameters(self) -> None:
        kwargs = dict(
            attractor_name="Lorenz",
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            zoom=1.6,
            time_value=0.0,
            luminosity=0.8,
        )
        with self.assertRaisesRegex(ValueError, "sample_count"):
            SnapshotRequest(**kwargs, sample_count=0)
        with self.assertRaisesRegex(ValueError, "burn_in"):
            SnapshotRequest(**kwargs, burn_in=-1)
        with self.assertRaisesRegex(ValueError, "sample_stride"):
            SnapshotRequest(**kwargs, sample_stride=0)

    def test_snapshot_filename_defaults_to_screenshot_folder(self) -> None:
        output = Path(snapshot_filename())
        self.assertEqual(output.parent, Path("screenshot"))
        self.assertTrue(output.name.startswith("attractor_"))
        self.assertEqual(output.suffix, ".png")

    def test_snapshot_controller_close_waits_for_active_export(self) -> None:
        controller = SnapshotController()
        finished = threading.Event()
        request = SnapshotRequest(
            attractor_name="Lorenz",
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            zoom=1.6,
            time_value=0.0,
            luminosity=0.8,
            sample_count=8,
            burn_in=1,
            sample_stride=1,
        )

        def fake_export(_request: SnapshotRequest) -> str:
            time.sleep(0.05)
            finished.set()
            return "/tmp/test_snapshot.png"

        with mock.patch("renderer.snapshot.export_attractor_snapshot", side_effect=fake_export):
            self.assertTrue(controller.start(request))
            controller.close()

        self.assertTrue(finished.is_set())
        self.assertFalse(controller.is_running)
        self.assertEqual(controller.message, "Snapshot saved: test_snapshot.png")


if __name__ == "__main__":
    unittest.main()
