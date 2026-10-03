import unittest

from casmi_ml.strong_slot_inference import select_strong_slots


class StrongSlotInferenceTests(unittest.TestCase):
    def test_explicit_original_pool_controls_every_slot_and_reference_position(self):
        original, current = ["a", "b", "c", "d"], ["a", "b", "c", "x", "y", "z", "d"]
        result = select_strong_slots(
            original,
            current,
            ["x", "y", "z"],
            {"a": 0.0, "x": 0.2, "y": 0.2, "z": 0.05},
            {"x": 4.0, "y": 3.0, "z": 5.0},
            {"a": 1.0, "b": 3.0, "c": 2.0},
            False,
            {"b"},
        )
        self.assertEqual(
            result, (["a", "b", "x", "c", "d"], ["x"], "high_strong_slot_inserted")
        )

    def test_missing_tied_budget_original_evidence_keeps163_ranking(self):
        original, current = ["a", "b"], ["a", "b", "x"]
        for pool, fallback in (
            ({"a": 1.0}, False),
            ({"a": 1.0, "b": 2.0}, False),
            ({"a": 0.0, "b": 0.0}, True),
        ):
            result, _, _ = select_strong_slots(
                original,
                current,
                ["x"],
                {"a": 0.0, "x": 0.2},
                {"x": 2.0},
                pool,
                fallback,
                set(),
            )
            self.assertEqual(result, current)
