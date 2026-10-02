import unittest

import numpy as np

from casmi_ml.critic_hard_negative_training import choose_hard


class CriticHardNegativeTests(unittest.TestCase):
    def test_scores_and_structure_keys_determine_hardness_without_labels(self):
        keys = ["Z", "B", "A", "C"]
        selected = choose_hard([0, 1, 2, 3], np.array([0.1, 0.7, 0.7, 0.2]), keys, 2)
        np.testing.assert_array_equal(selected, [2, 1])
        reordered = choose_hard([3, 2, 0, 1], np.array([0.2, 0.7, 0.1, 0.7]), keys, 2)
        np.testing.assert_array_equal(reordered, selected)

    def test_invalid_scored_pool_rejected(self):
        for pool, scores in [([0, 0], [1, 2]), ([0, 1], [1, np.nan]), ([0], [1])]:
            with self.assertRaises(ValueError):
                choose_hard(pool, np.asarray(scores), ["A", "B"], 2)
