from __future__ import annotations

import unittest

from config import FOG_RANGE, HELP_LINES, SPEED_RANGE
from hands.gestures import GestureInterpreter


def make_hand(
    wrist_x: float,
    wrist_y: float,
    *,
    index_mcp=(0.42, 0.44),
    middle_mcp=(0.5, 0.3),
    ring_mcp=(0.58, 0.44),
    pinky_mcp=(0.66, 0.50),
    thumb_tip=(0.52, 0.48),
    index_tip=(0.68, 0.28),
    ring_tip=(0.62, 0.34),
    pinky_tip=(0.78, 0.30),
):
    points = [(wrist_x, wrist_y, 0.0) for _ in range(21)]
    points[4] = (*thumb_tip, 0.0)
    points[5] = (*index_mcp, 0.0)
    points[8] = (*index_tip, 0.0)
    points[9] = (*middle_mcp, 0.0)
    points[13] = (*ring_mcp, 0.0)
    points[16] = (*ring_tip, 0.0)
    points[17] = (*pinky_mcp, 0.0)
    points[20] = (*pinky_tip, 0.0)
    return points


class GestureTests(unittest.TestCase):
    def test_help_lines_match_gesture_mapping(self) -> None:
        left_start = HELP_LINES.index("LEFT HAND") + 1
        left_end = HELP_LINES.index("", left_start)
        right_start = HELP_LINES.index("RIGHT HAND") + 1
        right_end = HELP_LINES.index("", right_start)
        keys_start = HELP_LINES.index("KEYS") + 1

        left_section = HELP_LINES[left_start:left_end]
        right_section = HELP_LINES[right_start:right_end]
        keys_section = HELP_LINES[keys_start:]

        self.assertIn("Thumb + index pinch -> Speed", left_section)
        self.assertIn("Thumb + ring pinch -> Luminosity", left_section)
        self.assertIn("Pinky touch palm -> Reset attractor", left_section)
        self.assertIn("Palm X -> Yaw", right_section)
        self.assertIn("Palm Y -> Pitch", right_section)
        self.assertIn("Thumb + index pinch -> Scale / zoom", right_section)
        self.assertIn("Thumb + ring pinch -> Fog", right_section)
        self.assertIn("Pinky touch palm -> Switch attractor", right_section)
        self.assertTrue(any("[ARROWS] rotate" in line for line in keys_section))
        self.assertTrue(any("[+/-] speed" in line for line in keys_section))
        self.assertTrue(any("[,/.] fog" in line for line in keys_section))
        self.assertTrue(any("[1-9] switch" in line for line in keys_section))

    def test_left_hand_controls_speed_and_ring_pinch_luminosity_while_right_controls_rotation_and_scale(self) -> None:
        interpreter = GestureInterpreter()
        left = make_hand(
            0.25,
            0.48,
            middle_mcp=(0.55, 0.75),
            thumb_tip=(0.30, 0.60),
            index_tip=(0.55, 0.60),
            ring_tip=(0.54, 0.60),
        )
        right = make_hand(0.65, 0.35, thumb_tip=(0.60, 0.40), index_tip=(0.80, 0.20))
        frame = interpreter.update({"left": left, "right": right}, now=1.0)
        self.assertGreater(frame.yaw, 0.0)
        self.assertLess(frame.pitch, 0.0)
        self.assertIsNotNone(frame.luminosity)
        self.assertGreater(frame.luminosity, 0.85)
        self.assertGreater(frame.scale, 0.3)
        self.assertIsNotNone(frame.fog)
        self.assertGreater(frame.fog, FOG_RANGE[0])
        self.assertAlmostEqual(frame.speed, SPEED_RANGE[1])
        self.assertFalse(frame.reset_current)
        self.assertEqual(frame.scene_delta, 0)
        self.assertFalse(hasattr(frame, "trail_len"))

    def test_left_hand_thumb_ring_pinch_controls_luminosity(self) -> None:
        interpreter = GestureInterpreter()
        bright = make_hand(
            0.35,
            0.40,
            thumb_tip=(0.30, 0.52),
            index_tip=(0.55, 0.52),
            ring_tip=(0.54, 0.52),
        )
        dim = make_hand(
            0.35,
            0.40,
            thumb_tip=(0.30, 0.52),
            index_tip=(0.55, 0.52),
            ring_tip=(0.31, 0.52),
        )

        bright_frame = interpreter.update({"left": bright, "right": None}, now=1.0)
        dim_frame = interpreter.update({"left": dim, "right": None}, now=1.1)

        self.assertIsNotNone(bright_frame.luminosity)
        self.assertIsNotNone(dim_frame.luminosity)
        self.assertGreater(bright_frame.luminosity, dim_frame.luminosity)
        self.assertAlmostEqual(bright_frame.speed, dim_frame.speed)

    def test_right_hand_thumb_ring_pinch_controls_fog(self) -> None:
        interpreter = GestureInterpreter()
        low_fog = make_hand(
            0.65,
            0.35,
            thumb_tip=(0.60, 0.40),
            index_tip=(0.80, 0.20),
            ring_tip=(0.61, 0.40),
        )
        high_fog = make_hand(
            0.65,
            0.35,
            thumb_tip=(0.60, 0.40),
            index_tip=(0.80, 0.20),
            ring_tip=(0.82, 0.40),
        )

        low_frame = interpreter.update({"left": None, "right": low_fog}, now=1.0)
        high_frame = interpreter.update({"left": None, "right": high_fog}, now=1.1)

        self.assertIsNotNone(low_frame.fog)
        self.assertIsNotNone(high_frame.fog)
        self.assertLess(low_frame.fog, high_frame.fog)
        self.assertGreaterEqual(low_frame.fog, FOG_RANGE[0])
        self.assertLessEqual(high_frame.fog, FOG_RANGE[1])
        self.assertAlmostEqual(low_frame.scale, high_frame.scale)
        self.assertFalse(hasattr(low_frame, "trail_len"))

    def test_left_pinky_resets_and_right_pinky_switches(self) -> None:
        interpreter = GestureInterpreter()
        left_trigger = make_hand(
            0.50,
            0.50,
            pinky_tip=(0.54, 0.46),
        )
        right_trigger = make_hand(
            0.50,
            0.50,
            pinky_tip=(0.54, 0.46),
        )
        release = make_hand(
            0.50,
            0.50,
            pinky_tip=(0.82, 0.26),
        )

        first_left = interpreter.update({"left": left_trigger, "right": None}, now=0.10)
        second_left = interpreter.update({"left": left_trigger, "right": None}, now=0.20)
        released_left = interpreter.update({"left": release, "right": None}, now=0.30)
        right_frame = interpreter.update({"left": None, "right": right_trigger}, now=0.90)

        self.assertTrue(first_left.reset_current)
        self.assertEqual(first_left.scene_delta, 0)
        self.assertFalse(second_left.reset_current)
        self.assertEqual(second_left.scene_delta, 0)
        self.assertFalse(released_left.reset_current)
        self.assertEqual(released_left.scene_delta, 0)
        self.assertFalse(right_frame.reset_current)
        self.assertEqual(right_frame.scene_delta, 1)

    def test_right_hand_motion_does_not_switch_scene(self) -> None:
        interpreter = GestureInterpreter()
        deltas = []
        for idx, wrist_x in enumerate((0.20, 0.25, 0.30, 0.36, 0.42)):
            right = make_hand(wrist_x, 0.40)
            frame = interpreter.update({"left": None, "right": right}, now=idx * 0.08)
            deltas.append(frame.scene_delta)
            self.assertIsNone(frame.speed)

        self.assertEqual(deltas, [0, 0, 0, 0, 0])

    def test_left_reset_rearms_when_pinky_leaves_palm(self) -> None:
        interpreter = GestureInterpreter()
        trigger = make_hand(
            0.50,
            0.50,
            pinky_tip=(0.54, 0.46),
        )
        release = make_hand(
            0.50,
            0.50,
            pinky_tip=(0.82, 0.26),
        )

        interpreter.update({"left": trigger, "right": None}, now=0.0)
        interpreter.update({"left": release, "right": None}, now=0.1)
        frame = interpreter.update({"left": trigger, "right": None}, now=0.7)
        self.assertTrue(frame.reset_current)
        self.assertEqual(frame.scene_delta, 0)

    def test_left_and_right_pinky_actions_have_independent_cooldowns(self) -> None:
        interpreter = GestureInterpreter()
        trigger = make_hand(0.50, 0.50, pinky_tip=(0.54, 0.46))
        release = make_hand(0.50, 0.50, pinky_tip=(0.82, 0.26))

        left_frame = interpreter.update({"left": trigger, "right": None}, now=0.10)
        interpreter.update({"left": release, "right": None}, now=0.20)
        right_frame = interpreter.update({"left": None, "right": trigger}, now=0.30)

        self.assertTrue(left_frame.reset_current)
        self.assertEqual(right_frame.scene_delta, 1)


if __name__ == "__main__":
    unittest.main()
