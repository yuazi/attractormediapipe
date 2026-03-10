from __future__ import annotations

import unittest

from hands.tracker import _assign_hand_slots, _normalize_handedness_label


def make_hand(center_x: float) -> list[tuple[float, float, float]]:
    points = [(center_x, 0.5, 0.0) for _ in range(21)]
    points[0] = (center_x, 0.7, 0.0)
    points[5] = (center_x - 0.03, 0.55, 0.0)
    points[9] = (center_x, 0.5, 0.0)
    points[13] = (center_x + 0.03, 0.55, 0.0)
    points[17] = (center_x + 0.06, 0.6, 0.0)
    return points


class TrackerTests(unittest.TestCase):
    def test_handedness_labels_stay_in_selfie_space(self) -> None:
        self.assertEqual(_normalize_handedness_label("Left"), "left")
        self.assertEqual(_normalize_handedness_label("Right"), "right")

    def test_unknown_handedness_label_is_ignored(self) -> None:
        self.assertIsNone(_normalize_handedness_label("unknown"))

    def test_two_hands_are_slotted_by_screen_position(self) -> None:
        rightish = make_hand(0.74)
        leftish = make_hand(0.21)
        slots = _assign_hand_slots([("right", rightish), ("right", leftish)])
        self.assertIs(slots["left"], leftish)
        self.assertIs(slots["right"], rightish)

    def test_two_uniquely_labeled_hands_keep_their_identity(self) -> None:
        crossed_left = make_hand(0.78)
        crossed_right = make_hand(0.22)
        slots = _assign_hand_slots([("left", crossed_left), ("right", crossed_right)])
        self.assertIs(slots["left"], crossed_left)
        self.assertIs(slots["right"], crossed_right)

    def test_unique_label_claims_slot_and_remaining_hand_fills_other_slot(self) -> None:
        labeled_left = make_hand(0.67)
        unlabeled_other = make_hand(0.31)
        slots = _assign_hand_slots([("left", labeled_left), (None, unlabeled_other)])
        self.assertIs(slots["left"], labeled_left)
        self.assertIs(slots["right"], unlabeled_other)

    def test_single_unknown_hand_falls_back_to_position(self) -> None:
        slots = _assign_hand_slots([(None, make_hand(0.77))])
        self.assertIsNotNone(slots["right"])
        self.assertIsNone(slots["left"])


if __name__ == "__main__":
    unittest.main()
