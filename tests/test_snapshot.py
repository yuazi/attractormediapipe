from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from config import SNAPSHOT_HEIGHT, SNAPSHOT_WIDTH
from renderer.snapshot import (
    SnapshotController,
    SnapshotRequest,
    _cover_frame_points,
    _render_density_image,
    ensure_snapshot_environment,
    export_attractor_snapshot,
    snapshot_filename,
)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_export_writes_png_at_requested_dimensions(self) -> None:
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
                width=640,
                height=360,
                sample_count=3000,
                burn_in=500,
                sample_stride=1,
            )
            result = export_attractor_snapshot(request)
            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertGreater(os.path.getsize(output_path), 0)
            with Image.open(output_path) as image:
                self.assertEqual(image.size, (640, 360))
                bbox = image.convert("L").point(lambda value: 255 if value > 0 else 0).getbbox()
                self.assertIsNotNone(bbox)
                assert bbox is not None
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                self.assertGreater(width, int(image.width * 0.65))
                self.assertGreater(height, int(image.height * 0.65))

    def test_cover_frame_points_scales_to_cover_snapshot_canvas(self) -> None:
        points = np.array(
            [
                [-0.30, -0.10],
                [0.30, -0.10],
                [-0.30, 0.10],
                [0.30, 0.10],
            ],
            dtype=np.float32,
        )

        covered = _cover_frame_points(points, overscan=1.0)

        self.assertLessEqual(float(np.min(covered[:, 1])), -1.0)
        self.assertGreaterEqual(float(np.max(covered[:, 1])), 1.0)
        self.assertGreater(float(np.max(np.abs(covered[:, 0]))), 1.0)

    def test_snapshot_request_defaults_to_configured_wallpaper_dimensions(self) -> None:
        request = SnapshotRequest(
            attractor_name="Lorenz",
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            zoom=1.6,
            time_value=0.0,
            luminosity=0.8,
        )

        self.assertEqual((request.width, request.height), (SNAPSHOT_WIDTH, SNAPSHOT_HEIGHT))

    def test_snapshot_request_rejects_invalid_dimensions_and_sampling_parameters(self) -> None:
        kwargs = dict(
            attractor_name="Lorenz",
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            zoom=1.6,
            time_value=0.0,
            luminosity=0.8,
        )
        with self.assertRaisesRegex(ValueError, "snapshot dimensions"):
            SnapshotRequest(**kwargs, width=0)
        with self.assertRaisesRegex(ValueError, "snapshot dimensions"):
            SnapshotRequest(**kwargs, height=0)
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

    def test_snapshot_density_palette_tracks_attractor_accent(self) -> None:
        density = np.array([[0.0, 1.0], [4.0, 12.0]], dtype=np.float32)
        warm = _render_density_image(density, 0.8, (230, 57, 70))
        cool = _render_density_image(density, 0.8, (17, 138, 178))

        warm_mean = warm[density > 0.0].mean(axis=0)
        cool_mean = cool[density > 0.0].mean(axis=0)

        self.assertGreater(warm_mean[0], warm_mean[2])
        self.assertGreater(cool_mean[2], cool_mean[0])

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
