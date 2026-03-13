from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageChops

from config import DEFAULT_FOG, SNAPSHOT_HEIGHT, SNAPSHOT_PRESET_NAMES, SNAPSHOT_WIDTH
from renderer.snapshot import (
    SnapshotController,
    SnapshotExportResult,
    SnapshotRequest,
    _cover_frame_points,
    _render_density_image,
    ensure_snapshot_environment,
    export_attractor_snapshot,
    snapshot_filename,
    snapshot_output_paths,
)
from screenshot import get_snapshot_preset


class SnapshotTests(unittest.TestCase):
    def test_snapshot_export_writes_png_at_requested_dimensions(self) -> None:
        ensure_snapshot_environment()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "snapshot.png"
            log_path = Path(tmpdir) / "snapshot_log.jsonl"
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
                preset_name="blueprint",
            )
            with mock.patch("renderer.snapshot.SNAPSHOT_LOG_PATH", str(log_path)):
                result = export_attractor_snapshot(request)
            self.assertEqual(result.clean_path, str(Path(tmpdir) / "snapshot_clean.png"))
            self.assertEqual(result.textured_path, str(Path(tmpdir) / "snapshot_textured.png"))
            self.assertTrue(Path(result.clean_path).exists())
            self.assertTrue(Path(result.textured_path).exists())
            self.assertGreater(os.path.getsize(result.clean_path), 0)
            self.assertGreater(os.path.getsize(result.textured_path), 0)
            with Image.open(result.clean_path) as image:
                self.assertEqual(image.size, (640, 360))
                bbox = image.convert("L").point(lambda value: 255 if value > 0 else 0).getbbox()
                self.assertIsNotNone(bbox)
                assert bbox is not None
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                self.assertGreater(width, int(image.width * 0.65))
                self.assertGreater(height, int(image.height * 0.65))
                self.assertEqual(image.info["Preset"], "blueprint")
                self.assertEqual(image.info["Attractor"], "Lorenz")
            with Image.open(result.clean_path) as clean_image, Image.open(result.textured_path) as textured_image:
                self.assertEqual(clean_image.size, (640, 360))
                self.assertEqual(textured_image.size, (640, 360))
                self.assertIsNone(ImageChops.difference(clean_image, textured_image).getbbox())

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
        self.assertEqual(request.fog, DEFAULT_FOG)

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
        with self.assertRaisesRegex(ValueError, "fog"):
            SnapshotRequest(**kwargs, fog=1.5)

    def test_snapshot_filename_defaults_to_screenshot_folder(self) -> None:
        output = Path(snapshot_filename())
        self.assertEqual(output.parent, Path("screenshot"))
        self.assertTrue(output.name.startswith("attractor_"))
        self.assertEqual(output.suffix, ".png")

    def test_snapshot_output_paths_add_variant_suffixes(self) -> None:
        output = snapshot_output_paths("wallpaper.png")
        self.assertEqual(output.clean, Path("wallpaper_clean.png"))
        self.assertEqual(output.textured, Path("wallpaper_textured.png"))

    def test_snapshot_density_palette_tracks_selected_preset(self) -> None:
        density = np.array([[0.0, 1.0], [4.0, 12.0]], dtype=np.float32)
        warm = _render_density_image(density, 0.8, "nebula")
        cool = _render_density_image(density, 0.8, "blueprint")

        warm_mean = warm[density > 0.0].mean(axis=0)
        cool_mean = cool[density > 0.0].mean(axis=0)

        self.assertGreater(warm_mean[0], warm_mean[2])
        self.assertGreater(cool_mean[2], cool_mean[0])

    def test_snapshot_preset_names_are_importable_from_config(self) -> None:
        self.assertEqual(set(SNAPSHOT_PRESET_NAMES), {"nebula", "blueprint", "void", "print"})
        for name in SNAPSHOT_PRESET_NAMES:
            self.assertEqual(get_snapshot_preset(name).name, name)

    def test_snapshot_export_embeds_metadata_and_appends_log(self) -> None:
        ensure_snapshot_environment()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metadata.png"
            log_path = Path(tmpdir) / "snapshot_log.jsonl"
            request = SnapshotRequest(
                attractor_name="Lorenz",
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                zoom=1.6,
                time_value=0.0,
                luminosity=0.8,
                output_path=str(output_path),
                width=320,
                height=180,
                sample_count=1500,
                burn_in=300,
                sample_stride=1,
                preset_name="void",
            )

            with mock.patch("renderer.snapshot.SNAPSHOT_LOG_PATH", str(log_path)):
                result = export_attractor_snapshot(request)

            with Image.open(result.clean_path) as image:
                self.assertEqual(image.info["Preset"], "void")
                self.assertEqual(image.info["Generator"], "attractormediapipe")
                self.assertIn("Parameters", image.info)
                self.assertEqual(image.info["Resolution"], "320x180")

            entries = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(entries), 1)
            self.assertIn('"Preset": "void"', entries[0])

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

        def fake_export(_request: SnapshotRequest) -> SnapshotExportResult:
            time.sleep(0.05)
            finished.set()
            return SnapshotExportResult("/tmp/test_snapshot_clean.png", "/tmp/test_snapshot_textured.png")

        with mock.patch("renderer.snapshot.export_attractor_snapshot", side_effect=fake_export):
            self.assertTrue(controller.start(request))
            controller.close()

        self.assertTrue(finished.is_set())
        self.assertFalse(controller.is_running)
        self.assertEqual(controller.message, "Snapshots saved: test_snapshot_clean.png, test_snapshot_textured.png")


if __name__ == "__main__":
    unittest.main()
