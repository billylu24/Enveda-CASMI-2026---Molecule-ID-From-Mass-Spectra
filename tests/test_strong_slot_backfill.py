import unittest

from casmi_ml.chembl_strong_slot_backfill import supported_promotion


class StrongSlotBackfillTests(unittest.TestCase):
    def test_deeper_candidate_only_fills_an_evidence_rejected_slot(self):
        result, changed = supported_promotion(
            ["a", "b", "c", "d"],
            ["a", "x", "b", "c", "d"],
            ["x", "bad", "y", "z", "unused"],
            {"a": 1, "b": 1, "c": 1},
            {"x": 5, "bad": 4, "y": 3, "z": 2, "unused": 1.5},
            False,
            {"b"},
            {"a": 0, "x": 0.2, "bad": 0.04, "y": 0.2, "z": 0.2, "unused": 0.3},
        )
        self.assertTrue(changed)
        self.assertEqual(result, ["a", "b", "x", "y", "z", "c", "d"])

    def test_tied_leader_or_incomplete_original_evidence_keeps181(self):
        for scores, fallback in [
            ({"a": 1, "b": 2}, False),
            ({"a": 1}, False),
            ({"a": 1, "b": 1}, True),
        ]:
            result, changed = supported_promotion(
                ["a", "b"],
                ["a", "b", "x"],
                ["x"],
                scores,
                {"x": 2},
                fallback,
                set(),
                {"a": 0, "x": 0.2},
            )
            self.assertFalse(changed)
            self.assertEqual(result, ["a", "b", "x"])
