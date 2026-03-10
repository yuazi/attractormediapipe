from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from renderer.snapshot import SnapshotRequest, ensure_snapshot_environment, export_attractor_snapshot


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


if __name__ == "__main__":
    unittest.main()
