import unittest

import numpy as np

from casmi_ml.chembl_fingerprint_prior import corrected_scores


class FingerprintPriorTests(unittest.TestCase):
    def test_equal_to_training_marginal_gives_no_candidate_preference(self):
        prior = np.linspace(0.01, 0.99, 2048)
        fps = np.stack([np.zeros(2048), np.ones(2048), np.arange(2048) % 2])
        np.testing.assert_allclose(corrected_scores(prior, fps, prior), 0, atol=1e-12)

    def test_invalid_training_prior_is_rejected(self):
        with self.assertRaises(ValueError):
            corrected_scores(np.ones(2048) * 0.5, np.zeros((1, 2048)), np.zeros(2048))
