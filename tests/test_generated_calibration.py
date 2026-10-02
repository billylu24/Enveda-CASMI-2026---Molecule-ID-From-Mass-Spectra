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


if __name__ == "__main__":
    unittest.main()
