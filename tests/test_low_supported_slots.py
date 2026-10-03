import unittest

from casmi_ml.low_supported_slots_evaluation import promote_supported_slots


class LowSupportedSlotsTests(unittest.TestCase):
    def test_each_slot_requires_both_gates_and_other_order_is_preserved(self):
        original = ["a", "b", "c", "d"]
        current = ["a", "b", "c", "x", "y", "z", "d"]
        result, supported = promote_supported_slots(
            original,
            current,
            ["x", "y", "z"],
            {"a": 0.4, "x": 0.6, "y": 0.46, "z": 0.45},
            {"a": 1, "b": 2, "c": 3, "x": 5, "y": 4, "z": 3.5},
            False,
        )
        self.assertEqual(supported, ["x", "y"])
        self.assertEqual(result, ["x", "y", "a", "b", "c", "z", "d"])

    def test_missing_prefix_or_budget_cannot_promote_any_slot(self):
        for fragments, fallback in [
            ({"a": 1, "x": 5, "y": 4}, False),
            ({"a": 1, "b": 2, "c": 3, "x": 5, "y": 4}, True),
        ]:
            result, supported = promote_supported_slots(
                ["a", "b", "c"],
                ["a", "b", "c", "x", "y"],
                ["x", "y"],
                {"a": 0.4, "x": 0.6, "y": 0.6},
                fragments,
                fallback,
            )
            self.assertEqual(result, ["a", "b", "c", "x", "y"])
            self.assertEqual(supported, [])
