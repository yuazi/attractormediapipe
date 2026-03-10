from __future__ import annotations

import unittest

from config import HELP_LINES, SPEED_RANGE
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
    pinky_tip=(0.78, 0.30),
):
    points = [(wrist_x, wrist_y, 0.0) for _ in range(21)]
    points[4] = (*thumb_tip, 0.0)
    points[5] = (*index_mcp, 0.0)
    points[8] = (*index_tip, 0.0)
    points[9] = (*middle_mcp, 0.0)
    points[13] = (*ring_mcp, 0.0)
    points[17] = (*pinky_mcp, 0.0)
    points[20] = (*pinky_tip, 0.0)
    return points


class GestureTests(unittest.TestCase):
    def test_help_lines_match_gesture_mapping(self) -> None:
        left_start = HELP_LINES.index("LEFT HAND") + 1
        left_end = HELP_LINES.index("", left_start)
        right_start = HELP_LINES.index("RIGHT HAND") + 1
        right_end = HELP_LINES.index("", right_start)

        left_section = HELP_LINES[left_start:left_end]
        right_section = HELP_LINES[right_start:right_end]

        self.assertIn("Palm X -> Yaw", left_section)
        self.assertIn("Palm Y -> Pitch", left_section)
        self.assertIn("Pinch -> Speed", left_section)
        self.assertIn("Pinky touch palm -> Previous attractor", left_section)
        self.assertIn("Index Y -> Luminosity", right_section)
        self.assertIn("Pinch -> Scale / zoom", right_section)
        self.assertIn("Pinky touch palm -> Next attractor", right_section)

    def test_left_pinch_controls_speed_and_right_maps_visuals(self) -> None:
        interpreter = GestureInterpreter()
        left = make_hand(
            0.25,
            0.75,
            middle_mcp=(0.55, 0.75),
            thumb_tip=(0.30, 0.60),
            index_tip=(0.55, 0.60),
        )
        right = make_hand(0.65, 0.35, thumb_tip=(0.60, 0.40), index_tip=(0.80, 0.20))
        frame = interpreter.update({"left": left, "right": right}, now=1.0)
        self.assertLess(frame.yaw, 0.0)
        self.assertGreater(frame.pitch, 0.0)
        self.assertIsNotNone(frame.luminosity)
        self.assertGreater(frame.luminosity, 0.05)
        self.assertGreater(frame.scale, 0.3)
        self.assertAlmostEqual(frame.speed, SPEED_RANGE[1])
        self.assertIsNone(frame.trail_len)
        self.assertEqual(frame.scene_delta, 0)

    def test_pinky_touch_switches_scene_by_hand(self) -> None:
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

        self.assertEqual(interpreter.update({"left": left_trigger, "right": None}, now=0.10).scene_delta, -1)
        self.assertEqual(interpreter.update({"left": left_trigger, "right": None}, now=0.20).scene_delta, 0)
        self.assertEqual(interpreter.update({"left": release, "right": None}, now=0.30).scene_delta, 0)
        self.assertEqual(interpreter.update({"left": None, "right": right_trigger}, now=0.90).scene_delta, 1)

    def test_right_hand_motion_does_not_switch_scene(self) -> None:
        interpreter = GestureInterpreter()
        deltas = []
        for idx, wrist_x in enumerate((0.20, 0.25, 0.30, 0.36, 0.42)):
            right = make_hand(wrist_x, 0.40)
            frame = interpreter.update({"left": None, "right": right}, now=idx * 0.08)
            deltas.append(frame.scene_delta)
            self.assertIsNone(frame.speed)

        self.assertEqual(deltas, [0, 0, 0, 0, 0])

    def test_scene_turn_rearms_when_pinky_leaves_palm(self) -> None:
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
        self.assertEqual(frame.scene_delta, -1)


if __name__ == "__main__":
    unittest.main()
