import unittest

import numpy as np

from casmi_ml.energy_conditioned_diagnostic_audit import (
    feature_block,
    fit_streaming_ridge,
)
from casmi_ml.forward_diagnostic_training_audit import fit_ridge


class EnergyConditionedAuditTest(unittest.TestCase):
    def test_structure_energy_interaction_changes_with_both_structure_and_energy(self):
        fps = np.array([[1, 0], [0, 1], [1, 0]], np.float32)
        energies = np.array([20, 20, 40])
        contexts = np.ones((3, 1))
        additive = feature_block(fps, energies, contexts)
        interaction = feature_block(fps, energies, contexts, True)
        np.testing.assert_array_equal(interaction[:, :4], additive)
        np.testing.assert_allclose(interaction[:, -2:], [[0.2, 0], [0, 0.2], [0.4, 0]])
        with self.assertRaises(ValueError):
            feature_block(fps, [-1, 20, 40], contexts)

    def test_streaming_matches_dense_fixed_ridge_and_excludes_holdout_targets(self):
        rng = np.random.default_rng(32)
        features = rng.normal(size=(19, 4))
        targets = rng.random((19, 3))
        fit = np.arange(19) < 13
        block = lambda ids: features[ids]
        predicted, model = fit_streaming_ridge(
            block, targets, fit, device="cpu", batch=5
        )
        reference, dense = fit_ridge(features, targets, fit, device="cpu")
        np.testing.assert_allclose(predicted, reference, atol=1e-12)
        for name in model:
            np.testing.assert_allclose(model[name], dense[name], atol=1e-12)
        modified = targets.copy()
        modified[~fit] = 100
        other, new = fit_streaming_ridge(block, modified, fit, device="cpu", batch=7)
        np.testing.assert_allclose(other, predicted, atol=1e-12)
        for name in model:
            np.testing.assert_allclose(model[name], new[name], atol=1e-12)

    def test_no_fit_rows_rejected(self):
        with self.assertRaises(ValueError):
            fit_streaming_ridge(
                lambda ids: np.zeros((len(ids), 4)),
                np.zeros((2, 3)),
                [False, False],
                device="cpu",
            )
