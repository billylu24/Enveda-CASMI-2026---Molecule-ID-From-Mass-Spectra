import unittest

import numpy as np

from casmi_ml.mass_candidate_capacity_audit import fit_mass_bonus


class MassCandidateCapacityTest(unittest.TestCase):
    def test_excluded_negative_cannot_drive_chemistry_gradient(self):
        cos = np.zeros((2, 16))
        features = np.zeros((2, 16, 1))
        features[:, 0] = 1
        allowed = np.zeros((2, 15), bool)
        weights, report = fit_mass_bonus(cos, features, 5, allowed)
        np.testing.assert_allclose(weights, 0, atol=1e-10)
        self.assertEqual(report["final_objective"], 0.0)
        self.assertEqual(report["compatible_pairs"], 0)

    def test_compatible_competitor_can_use_positive_evidence(self):
        cos = np.zeros((2, 16))
        features = np.zeros((2, 16, 1))
        features[:, 0] = 1
        allowed = np.zeros((2, 15), bool)
        allowed[:, 0] = True
        weights, report = fit_mass_bonus(cos, features, 5, allowed)
        self.assertGreater(weights[0], 0)
        self.assertLessEqual(weights.sum(), 2)
        self.assertGreater(report["same_mask_objective_gain"], 0)
