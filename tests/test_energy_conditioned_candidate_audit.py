import unittest

import numpy as np

from casmi_ml.energy_conditioned_candidate_audit import candidate_prediction


class ConditionalCandidateTest(unittest.TestCase):
    def test_candidate_comparison_uses_same_query_condition(self):
        fps = np.array([[1, 0], [0, 1]], np.float32)
        model = {
            "weights": np.zeros((6, 1)),
            "mean_x": np.zeros(6),
            "mean_y": np.zeros(1),
        }
        model["weights"][-2, 0] = 0.5
        low = candidate_prediction(fps, [20, 20], np.ones((2, 1)), model, True)
        high = candidate_prediction(fps, [40, 40], np.ones((2, 1)), model, True)
        np.testing.assert_allclose(low[:, 0], [0.1, 0])
        np.testing.assert_allclose(high[:, 0], [0.2, 0])

    def test_outside_strength_range_clamped(self):
        model = {
            "weights": np.ones((3, 1)) * 20,
            "mean_x": np.zeros(3),
            "mean_y": np.zeros(1),
        }
        actual = candidate_prediction(
            np.ones((1, 1)), [20], np.ones((1, 1)), model, False
        )
        np.testing.assert_array_equal(actual, [[1]])
