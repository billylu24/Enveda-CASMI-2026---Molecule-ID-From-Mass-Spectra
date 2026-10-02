import unittest

import numpy as np

from casmi_ml.generated_spectrum_critic import aggregate


class SpectrumCriticTests(unittest.TestCase):
    def test_aggregation_distinguishes_spectral_evidence_and_preserves_permutation(
        self,
    ):
        scores = np.array([[0.9, 0.4], [0.1, 0.5], [0.1, 0.6]])
        self.assertGreater(aggregate(scores, "max")[0], aggregate(scores, "max")[1])
        self.assertLess(
            aggregate(scores, "baseline")[0], aggregate(scores, "baseline")[1]
        )
        np.testing.assert_allclose(aggregate(scores, "top2"), [0.5, 0.55])
        for mode in ["baseline", "max", "median", "top2"]:
            np.testing.assert_equal(
                aggregate(scores, mode), aggregate(scores[::-1], mode)
            )
            np.testing.assert_equal(aggregate(scores[:1], mode), scores[0])
        with self.assertRaises(ValueError):
            aggregate(np.array([[float("nan")]]), "baseline")
