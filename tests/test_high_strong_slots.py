import unittest

from casmi_ml.chembl_high_strong_slots import supported_promotion


class StrongSlotEvidenceTests(unittest.TestCase):
    def test_each_slot_requires_all_original_fragment_and_critic_support(self):
        prior, current = ["a", "b", "c", "d"], ["a", "b", "c", "x", "y", "z", "d"]
        result, moved = supported_promotion(
            prior,
            current,
            ["x", "y", "z", "other"],
            {"a": 1.0, "b": 3.0, "c": 2.0},
            {"x": 4.0, "y": 3.0, "z": 5.0},
            False,
            {"b"},
            {"a": 0.0, "x": 0.1, "y": 0.1, "z": 0.05},
        )
        self.assertEqual(result, ["a", "b", "x", "c", "d"])
        self.assertTrue(moved)

    def test_leader_failure_or_budget_keeps163_whole_ranking(self):
        prior, current = ["a", "b"], ["a", "b", "x"]
        for fragment, fallback in ((2.0, False), (3.0, True)):
            self.assertEqual(
                supported_promotion(
                    prior,
                    current,
                    ["x"],
                    {"a": 1.0, "b": 2.0},
                    {"x": fragment},
                    fallback,
                    set(),
                    {"a": 0.0, "x": 0.2},
                ),
                (current, False),
            )
