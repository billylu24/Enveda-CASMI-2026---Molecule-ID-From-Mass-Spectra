import unittest

import numpy as np

from casmi_ml.candidate_array_order import rank_values


class ArrayCandidateOrderTests(unittest.TestCase):
    def test_ties_signed_zeros_and_close_scores_match_scalar_reference(self):
        keys = ["z", "a", "b", "c", "x"]
        values = np.array([0.0, -0.0, 1.0, np.nextafter(1.0, 2.0), -1.0])
        expected = [
            keys[i]
            for i in sorted(range(len(keys)), key=lambda i: (-values[i], keys[i]))
        ]
        self.assertEqual(rank_values(keys, values), expected)
        self.assertEqual(rank_values([], []), [])

    def test_randomized_finite_float32_scores_match_all_scalar_orders(self):
        rng = np.random.default_rng(42)
        for count in [1, 20, 1000]:
            keys = [str(i).zfill(14) for i in rng.permutation(count)]
            values = rng.integers(-20, 20, count).astype(np.float32)
            expected = [
                keys[i]
                for i in sorted(range(count), key=lambda i: (-values[i], keys[i]))
            ]
            self.assertEqual(rank_values(keys, values), expected)

    def test_nonfinite_dimension_or_duplicate_keys_are_rejected(self):
        for keys, values in [
            (["a"], [float("nan")]),
            (["a"], [1, 2]),
            (["a", "a"], [1, 2]),
        ]:
            with self.assertRaises(ValueError):
                rank_values(keys, values)
