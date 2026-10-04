import unittest

import numpy as np

from casmi_ml.energy_conditioned_candidate_audit import candidate_prediction
from casmi_ml.energy_conditioned_capacity_audit import conditional_interactions


class ConditionedCapacityTest(unittest.TestCase):
    def test_factorized_same_query_prediction_matches_full_feature_graph(self):
        rng = np.random.default_rng(826)
        fps = rng.integers(0, 2, (2, 2048)).astype(np.float64)
        contexts = rng.random((2, 2, 3))
        energies = np.array([[20.0, 40.0], [30.0, 50.0]])
        observed = rng.random((2, 2, 60))
        weights = rng.normal(0, 0.005, (4100, 60))
        model = {
            "weights": weights,
            "mean_x": rng.random(4100),
            "mean_y": rng.random(60),
        }
        selected = np.array([[0] + [1] * 15, [1] + [0] * 15])
        projections = (fps @ weights[:2048], fps @ weights[-2048:])
        actual = conditional_interactions(
            observed, energies, contexts, projections, model, selected
        )
        expected = np.zeros_like(actual)
        for q in range(2):
            for slot in range(2):
                predicted = candidate_prediction(
                    fps[selected[q]],
                    np.repeat(energies[q, slot], 16),
                    np.repeat(contexts[q, slot][None], 16, axis=0),
                    model,
                    True,
                )
                expected[q] += observed[q, slot] * predicted / 2
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_missing_or_merged_energy_contributes_zero_extra_and_keeps_pair_denominator(
        self,
    ):
        observed = np.ones((2, 2, 60))
        energies = np.array([[np.nan, np.nan], [20.0, np.nan]])
        contexts = np.zeros((2, 2, 1))
        model = {
            "weights": np.zeros((4098, 60)),
            "mean_x": np.zeros(4098),
            "mean_y": np.ones(60),
        }
        projections = (np.zeros((2, 60)), np.zeros((2, 60)))
        selected = np.zeros((2, 16), int)
        actual = conditional_interactions(
            observed, energies, contexts, projections, model, selected
        )
        np.testing.assert_array_equal(actual[0], np.zeros((16, 60)))
        np.testing.assert_array_equal(actual[1], np.ones((16, 60)) * 0.5)

    def test_unobserved_diagnostics_never_add_candidate_bonus(self):
        observed = np.zeros((2, 2, 60))
        energies = np.ones((2, 2)) * 20
        model = {
            "weights": np.zeros((4098, 60)),
            "mean_x": np.zeros(4098),
            "mean_y": np.ones(60),
        }
        actual = conditional_interactions(
            observed,
            energies,
            np.zeros((2, 2, 1)),
            (np.zeros((2, 60)), np.zeros((2, 60))),
            model,
            np.zeros((2, 16), int),
        )
        self.assertFalse(actual.any())
