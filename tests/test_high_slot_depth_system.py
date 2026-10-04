import unittest

from casmi_ml.high_slot_depth_system import select_slots
from casmi_ml.strong_slot_inference import select_strong_slots


class HighSlotDepthTest(unittest.TestCase):
    def setUp(self):
        self.args = (
            ["a", "b", "c", "d"],
            ["a", "b", "c", "x", "y", "z", "d"],
            ["x", "y", "z", "v", "w", "u"],
            {"a": 0.5, "x": 0.7, "y": 0.8, "z": 0.9, "v": 0.6, "w": 0.4, "u": 0.9},
            {"x": 10.0, "y": 8.0, "z": 2.0, "v": 7.0, "w": 9.0, "u": 11.0},
            {"a": 3.0, "b": 4.0, "c": 5.0},
            False,
            {"a", "b"},
        )

    def test_three_is_exact(self):
        self.assertEqual(
            select_slots(*self.args, slots=3), select_strong_slots(*self.args)
        )

    def test_individual_gates_and_depth(self):
        ranking, supported, status = select_slots(*self.args, slots=5)
        self.assertEqual(supported, ["x", "y", "v"])
        self.assertEqual(ranking, ["a", "b", "x", "y", "v", "c", "d"])
        self.assertEqual(status, "high_strong_slot_inserted")

    def test_leader_cannot_be_rescued_by_deeper_candidate(self):
        args = list(self.args)
        args[4] = {**args[4], "x": 5.0}
        self.assertEqual(select_slots(*args, slots=5)[0], args[1])

    def test_missing_original_evidence_preserves_fallback(self):
        args = list(self.args)
        args[5] = {"a": 3.0}
        self.assertEqual(select_slots(*args, slots=5)[0], args[1])

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            select_slots(*self.args, slots=4)
