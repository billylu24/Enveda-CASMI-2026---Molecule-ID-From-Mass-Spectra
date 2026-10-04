import unittest

from casmi_ml.individual_gate_system import select_high, select_strong


class IndividualGateTest(unittest.TestCase):
    def test_rejects_unqualified_leader_but_accepts_supported_candidate(self):
        result, inserted, status = select_high(
            ["a", "b", "c"],
            ["a", "b", "c", "x", "y"],
            ["x", "y"],
            {"a": 0.5, "x": 0.9, "y": 0.7},
            {"a": 2.0, "x": 1.0, "y": 10.0},
            False,
        )
        self.assertEqual(inserted, ["y"])
        self.assertEqual(result, ["a", "b", "c", "y"])
        self.assertEqual(status, "high_fragment_inserted")

    def test_missing_or_tied_physics_preserves_tail(self):
        original = ["a", "b", "c"]
        tail = original + ["x"]
        self.assertEqual(
            select_high(
                original, tail, ["x"], {"a": 0.5, "x": 0.9}, {"a": 0.0, "x": 0.0}, False
            )[0],
            tail,
        )

    def test_strong_slots_filter_all_before_capping(self):
        original = ["a", "b", "c", "d"]
        proposed = ["x", "y", "z", "v", "w"]
        result, supported, _ = select_strong(
            original,
            original,
            proposed,
            {"a": 0.5, "x": 0.9, "y": 0.9, "z": 0.5, "v": 0.8, "w": 0.8},
            {"x": 2.0, "y": 8.0, "z": 9.0, "v": 10.0, "w": 11.0},
            {"a": 3.0, "b": 4.0, "c": 5.0},
            False,
            {"a", "b"},
        )
        self.assertEqual(supported, ["y", "v", "w"])
        self.assertEqual(result, ["a", "b", "y", "v", "w", "c", "d"])

    def test_original_evidence_fallback_keeps_previous_result(self):
        self.assertEqual(
            select_strong(
                ["a", "b"],
                ["a", "x", "b"],
                ["x"],
                {"a": 0.1, "x": 0.9},
                {"x": 10.0},
                {"a": 1.0},
                False,
                {"a"},
            )[0],
            ["a", "x", "b"],
        )
