import unittest

from casmi_ml.generated_calibration import calibrated_order


class GeneratedCalibrationTests(unittest.TestCase):
    def test_length_calibration_can_remove_short_smiles_preference(self):
        candidates = [
            {
                "key": "A",
                "smiles": "CC",
                "log_probability": -2,
                "chemical_score": 0,
                "formula_support": 0,
            },
            {
                "key": "B",
                "smiles": "CCCCCCCC",
                "log_probability": -4,
                "chemical_score": 0,
                "formula_support": 0,
            },
        ]
        self.assertEqual(calibrated_order(candidates, 0), ["A", "B"])
        self.assertEqual(calibrated_order(candidates, 1), ["B", "A"])
        with self.assertRaises(ValueError):
            calibrated_order(candidates, -1)

    def test_control_preserves_frozen_chemical_order(self):
        candidates = [{"key": "B"}, {"key": "A"}]
        self.assertEqual(calibrated_order(candidates, 0), ["B", "A"])
        self.assertEqual(calibrated_order([], 1), [])

    def test_measured_sequence_length_is_distinct_from_canonical_characters(self):
        candidates = [
            {
                "key": "A",
                "smiles": "CC",
                "log_probability": -3,
                "best_sequence_tokens": 9,
                "chemical_score": 0,
                "formula_support": 0,
            },
            {
                "key": "B",
                "smiles": "CCCC",
                "log_probability": -4,
                "best_sequence_tokens": 4,
                "chemical_score": 0,
                "formula_support": 0,
            },
        ]
        self.assertEqual(calibrated_order(candidates, 1), ["B", "A"])
        self.assertEqual(calibrated_order(candidates, 1, "sampled_tokens"), ["A", "B"])
        with self.assertRaises(ValueError):
            calibrated_order(
                [
                    {k: v for k, v in c.items() if k != "best_sequence_tokens"}
                    for c in candidates
                ],
                1,
                "sampled_tokens",
            )

    def test_weak_prior_weights_can_be_disabled_without_losing_length_order(self):
        candidates = [
            {
                "key": "A",
                "smiles": "CC",
                "log_probability": -2,
                "best_sequence_tokens": 2,
                "chemical_score": 0,
                "formula_support": 0,
            },
            {
                "key": "B",
                "smiles": "CO",
                "log_probability": -4,
                "best_sequence_tokens": 2,
                "chemical_score": 0,
                "formula_support": 1,
            },
        ]
        self.assertEqual(
            calibrated_order(candidates, 1, "sampled_tokens", 0, 1), ["B", "A"]
        )
        self.assertEqual(
            calibrated_order(candidates, 1, "sampled_tokens", 0, 0), ["A", "B"]
        )
        with self.assertRaises(ValueError):
            calibrated_order(candidates, 1, "sampled_tokens", 0, float("nan"))


if __name__ == "__main__":
    unittest.main()
