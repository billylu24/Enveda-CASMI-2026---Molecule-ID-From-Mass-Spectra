import unittest

import numpy as np

from casmi_ml.forward_candidate_capacity_audit import predicted_interactions


class ForwardCandidateCapacityTest(unittest.TestCase):
    def test_no_observed_peak_cannot_create_candidate_bonus(self):
        chemical = np.zeros((2, 60))
        predictions = np.ones((2, 60))
        selected = np.zeros((2, 16), np.int64)
        np.testing.assert_array_equal(
            predicted_interactions(chemical, predictions, selected), 0
        )
        chemical[0, 4] = 0.5
        predictions[1, 4] = 0.25
        selected[0, 2] = 1
        actual = predicted_interactions(chemical, predictions, selected)
        self.assertEqual(actual[0, 2, 4], 0.125)
        self.assertEqual(actual[1].sum(), 0)

    def test_invalid_strength_or_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            predicted_interactions(
                np.ones((2, 60)), np.ones((2, 60)) * 2, np.zeros((2, 16), np.int64)
            )
        with self.assertRaises(ValueError):
            predicted_interactions(
                np.ones((2, 60)), np.ones((2, 60)), np.full((2, 16), -1)
            )
