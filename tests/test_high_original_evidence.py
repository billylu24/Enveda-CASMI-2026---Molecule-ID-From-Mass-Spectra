import unittest

from casmi_ml.chembl_high_original_evidence import supported_promotion


class OriginalFragmentEvidenceTests(unittest.TestCase):
    def test_supported_original_second_prevents_displacement(self):
        prior = ["a", "b", "c", "d"]
        current = ["a", "b", "c", "x", "y", "z", "d"]
        result, moved = supported_promotion(
            prior, current, ["x", "y", "z"], {"a": 1, "b": 3, "c": 2}, {"x": 2.5}, False
        )
        self.assertEqual(result, current)
        self.assertFalse(moved)
        result, moved = supported_promotion(
            prior, current, ["x", "y", "z"], {"a": 1, "b": 3, "c": 2}, {"x": 4}, False
        )
        self.assertEqual(result, ["a", "x", "y", "z", "b", "c", "d"])
        self.assertTrue(moved)

    def test_missing_tied_or_budget_evidence_preserves163(self):
        prior, current = ["a", "b"], ["a", "b", "x"]
        for scores, budget in [
            ({"a": 1}, False),
            ({"a": 1, "b": 2}, False),
            ({"a": 1, "b": 0}, True),
        ]:
            self.assertEqual(
                supported_promotion(prior, current, ["x"], scores, {"x": 2}, budget),
                (current, False),
            )
