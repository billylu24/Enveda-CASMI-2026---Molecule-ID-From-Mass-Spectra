import unittest

import numpy as np

from casmi_ml.chembl_mass_prior import local_prior


class MassConditionedPriorTests(unittest.TestCase):
    def test_nearest_masses_equal_weight_and_laplace_smoothing(self):
        masses = np.array([10.0, 20.0, 30.0, 40.0])
        keys = np.array(["a", "b", "c", "d"])
        fps = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        np.testing.assert_allclose(local_prior(26.0, masses, keys, fps, 2), [0.75, 0.5])

    def test_equal_distances_resolve_by_key(self):
        np.testing.assert_allclose(
            local_prior(
                25.0,
                np.array([20.0, 30.0]),
                np.array(["z", "a"]),
                np.array([[0], [1]]),
                1,
            ),
            [2 / 3],
        )

    def test_mass_boundary_tie_includes_entire_duplicate_mass_run(self):
        masses = np.full(8, 30.0)
        keys = np.array(["z", "y", "x", "w", "v", "u", "b", "a"])
        fps = np.array([[0], [0], [0], [0], [0], [0], [1], [1]])
        np.testing.assert_allclose(local_prior(30.0, masses, keys, fps, 2), [0.75])

    def test_nonfinite_observable_mass_rejected(self):
        with self.assertRaises(ValueError):
            local_prior(
                float("nan"), np.array([20.0]), np.array(["a"]), np.array([[1]]), 1
            )
