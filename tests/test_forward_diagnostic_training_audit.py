import unittest

import numpy as np

from casmi_ml.forward_diagnostic_training_audit import fit_ridge, heldout_mask


class ForwardDiagnosticTrainingTest(unittest.TestCase):
    def test_heldout_targets_cannot_change_fitted_weights_or_intercept(self):
        features = np.asarray([[0], [1], [0.5], [0.2]], float)
        targets = features.copy()
        fit = np.asarray([True, True, False, False])
        predicted, model = fit_ridge(features, targets, fit, device="cpu")
        changed = targets.copy()
        changed[~fit] = 100
        new_predicted, new_model = fit_ridge(features, changed, fit, device="cpu")
        np.testing.assert_array_equal(predicted, new_predicted)
        for name in model:
            np.testing.assert_array_equal(model[name], new_model[name])
        self.assertLess(np.square(predicted[~fit] - targets[~fit]).mean(), 0.001)

    def test_molecular_hash_assignment_preserves_identity_and_order(self):
        keys = ["A", "B", "C", "A"]
        original = heldout_mask(keys)
        self.assertEqual(original[0], original[3])
        np.testing.assert_array_equal(heldout_mask(keys[::-1]), original[::-1])
