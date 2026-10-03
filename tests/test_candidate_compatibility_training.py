import unittest

import numpy as np

from casmi_ml.candidate_compatibility_training import fit_bonus


class CandidateBonusTrainingTests(unittest.TestCase):
    def test_learns_positive_structural_interaction_under_total_budget(self):
        scores = np.zeros((12, 16))
        features = np.zeros((12, 16, 24))
        features[:, 0, 4] = 1
        features[:, 0, 7] = 0.5
        weights, report = fit_bonus(scores, features, 10)
        self.assertGreater(weights[4], 0.049)
        self.assertLessEqual(weights.sum(), 0.05)
        self.assertTrue((weights >= 0).all())
        self.assertLess(report["final_objective"], report["initial_ce"])
        masked, control = fit_bonus(scores, features, 10, mask=True)
        np.testing.assert_array_equal(masked, np.zeros(24))
        self.assertEqual(control["final_objective"], control["initial_ce"])

    def test_rejects_out_of_bound_inputs_and_expired_budget(self):
        scores = np.zeros((2, 16))
        features = np.zeros((2, 16, 24))
        features[0, 0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            fit_bonus(scores, features, 10)
        features.fill(0)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            fit_bonus(scores, features, 10, checkpoint=lambda: False)
