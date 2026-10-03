import unittest

import numpy as np
import pandas as pd
import torch

from casmi_ml.training_query_groups import grouped_rows, masked_group_mean


class TrainingQueryGroupTests(unittest.TestCase):
    def test_all_measured_rows_retained_across_ions_and_padding_has_zero_vote(self):
        frame = pd.DataFrame(
            {
                "inchikey14": ["b", "a", "a", "a", "c", "b"],
                "adduct": ["H", "H", "Na", "minusH", "H", "minusH"],
            }
        )
        indexes, valid = grouped_rows(frame, np.array([2, 5, 4]))
        np.testing.assert_array_equal(indexes, [[1, 2, 3], [0, 5, 5], [4, 4, 4]])
        np.testing.assert_array_equal(
            valid, [[True, True, True], [True, True, False], [True, False, False]]
        )
        values = torch.tensor(
            [
                [[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]],
                [[2.0, 6.0], [4.0, 2.0], [999.0, -999.0]],
                [[3.0, 4.0], [999.0, 999.0], [-999.0, -999.0]],
            ]
        )
        actual = masked_group_mean(values, torch.from_numpy(valid))
        torch.testing.assert_close(
            actual, torch.tensor([[2.0, 4.0], [3.0, 4.0], [3.0, 4.0]]), rtol=0, atol=0
        )
        with self.assertRaisesRegex(ValueError, "another molecule"):
            grouped_rows(frame, [0, 5, 4])
        with self.assertRaisesRegex(ValueError, "valid row"):
            masked_group_mean(values, torch.zeros_like(torch.from_numpy(valid)))
