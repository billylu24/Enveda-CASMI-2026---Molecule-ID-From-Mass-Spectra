import unittest

import numpy as np

from casmi_ml.candidate_chemistry_specificity_audit import contrast_summary


class SpecificitySummaryTests(unittest.TestCase):
    def test_mass_subset_excludes_unavailable_comparisons_and_keeps_ties(self):
        result = contrast_summary(
            np.array([0.3, 0.0]),
            np.array([[0.2, 0.3, 0.4], [0.0, 0.1, 0.0]]),
            np.array([[True, False, True], [False, True, True]]),
        )
        self.assertEqual(result["unrestricted"]["pairs"], 6)
        self.assertEqual(result["unrestricted"]["bonus_ties"], 3)
        self.assertEqual(result["mass_compatible"]["pairs"], 4)
        self.assertEqual(result["mass_compatible"]["positive_bonus_larger"], 1)
        self.assertEqual(result["mass_compatible"]["negative_bonus_larger"], 2)
        self.assertEqual(result["mass_compatible"]["bonus_ties"], 1)
        self.assertEqual(
            result["mass_compatible"]["molecules_with_any_positive_advantage"], 1
        )

    def test_empty_mass_subset_and_invalid_scores(self):
        result = contrast_summary([0.0], [[0.0]], [[False]])
        self.assertEqual(result["mass_compatible"]["pairs"], 0)
        self.assertIsNone(result["mass_compatible"]["mean_positive_minus_negative"])
        with self.assertRaises(ValueError):
            contrast_summary([float("nan")], [[0.0]], [[True]])
        with self.assertRaises(ValueError):
            contrast_summary([0.0], [[0.0]], [[True, False]])


if __name__ == "__main__":
    unittest.main()
