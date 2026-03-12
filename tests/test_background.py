from __future__ import annotations

import unittest

import numpy as np

from renderer.background import get_background_layers


class BackgroundTests(unittest.TestCase):
    def test_background_texture_covers_full_frame_with_points(self) -> None:
        texture = get_background_layers(320, 180, (255, 88, 88)).texture_rgba[:, :, :3].astype(np.float32)
        brightness = texture.mean(axis=2)

        top_density = float((brightness[:60] > 10.0).mean())
        middle_density = float((brightness[60:120] > 10.0).mean())
        lower_density = float((brightness[120:] > 10.0).mean())

        self.assertGreater(top_density, 0.02)
        self.assertGreater(middle_density, 0.05)
        self.assertGreater(lower_density, 0.05)

    def test_background_texture_keeps_lattice_visibility(self) -> None:
        texture = get_background_layers(320, 180, (255, 88, 88)).texture_rgba[:, :, :3].astype(np.float32)
        brightness = texture.mean(axis=2)

        row_profile = (brightness[90] > 10.0).astype(np.float32)
        transitions = float(np.abs(np.diff(row_profile)).sum())

        self.assertGreater(transitions, 30.0)

    def test_background_fog_stays_low_in_the_frame(self) -> None:
        fog = get_background_layers(320, 180, (98, 255, 229)).fog_rgba[:, :, 3].astype(np.float32)

        top_mean = float(fog[:60].mean())
        lower_mean = float(fog[100:].mean())

        self.assertGreater(lower_mean, top_mean + 6.0)


if __name__ == "__main__":
    unittest.main()
