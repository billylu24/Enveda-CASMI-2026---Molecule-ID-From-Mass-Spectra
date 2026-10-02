import unittest

import numpy as np

from casmi_ml.generated_native_scores import native_scores
from casmi_ml.ranking import neural_rank


class NativeScoresTests(unittest.TestCase):
    def test_same_bernoulli_score_as_deployed_native_ranking(self):
        probability = np.full(2048, 0.1)
        probability[1] = 0.9
        fps = np.zeros((2, 2048))
        fps[0, 0] = 1
        fps[1, 1] = 1
        scores = native_scores(probability, fps)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(neural_rank(probability, ["a", "b"], fps), ["b", "a"])
        np.testing.assert_allclose(native_scores(np.full(2048, 0.5), fps), [0, 0])
        with self.assertRaises(ValueError):
            native_scores(probability[:10], fps)
