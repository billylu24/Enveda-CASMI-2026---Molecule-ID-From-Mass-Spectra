import unittest

import numpy as np

from casmi_ml.strict_mass_negative_mask import admissible_negatives


class StrictMassNegativeTests(unittest.TestCase):
    def test_exact_same_window_floor_and_ppm_without_positive_leak(self):
        masses = np.array([100.0, 100.005, 100.02, 1000.0, 1000.02])
        hard = np.array([[1, 2], [0, 2], [0, 1], [4, 0], [3, 0]])
        np.testing.assert_array_equal(
            admissible_negatives(masses, hard),
            [
                [True, False],
                [True, False],
                [False, False],
                [True, False],
                [True, False],
            ],
        )

    def test_self_negative_or_bad_mass_or_outside_index_is_rejected(self):
        for masses, hard in [
            ([100.0, 100.0], [[0], [0]]),
            ([float("nan"), 100.0], [[1], [0]]),
            ([100.0, 100.0], [[2], [0]]),
        ]:
            with self.assertRaises(ValueError):
                admissible_negatives(masses, np.array(hard))
