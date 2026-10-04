import unittest

import numpy as np

from casmi_ml.diagnostic_bonus_generalization_audit import (
    candidate_losses,
    molecular_bootstrap_gain,
)


class BonusGeneralizationAuditTest(unittest.TestCase):
    def test_restricted_denominator_retains_positive_and_ignores_excluded_negative(
        self,
    ):
        scores = np.zeros((2, 16))
        allowed = np.zeros((2, 16), bool)
        allowed[:, 0] = True
        allowed[0, 1] = True
        scores[:, 2] = 1e6
        loss = candidate_losses(scores, 10, allowed)
        np.testing.assert_allclose(loss, [np.log(2), 0])
        allowed[1, 0] = False
        with self.assertRaises(ValueError):
            candidate_losses(scores, 10, allowed)

    def test_paired_molecule_bootstrap_identity_and_constant_reduction(self):
        x = np.array([1.0, 2.0, 4.0])
        identity = molecular_bootstrap_gain(x, x, draws=100)
        self.assertEqual(identity["mean_loss_reduction"], 0)
        self.assertEqual(identity["paired_bootstrap95_loss_reduction"], [0, 0])
        reduced = molecular_bootstrap_gain(x, x - 0.25, draws=100)
        np.testing.assert_allclose(
            reduced["paired_bootstrap95_loss_reduction"], [0.25, 0.25]
        )
        self.assertEqual(reduced, molecular_bootstrap_gain(x, x - 0.25, draws=100))


if __name__ == "__main__":
    unittest.main()
