import unittest

import numpy as np
import pandas as pd

from casmi_ml.training_query_pairs import paired_rows


class TrainingQueryPairTests(unittest.TestCase):
    def test_second_row_is_distinct_same_molecule_adduct_and_mode(self):
        frame = pd.DataFrame(
            {
                "inchikey14": ["a", "a", "a", "a", "b"],
                "adduct": ["[M+H]+", "[M+H]+", "[M+Na]+", "[M+H]+", "[M-H]-"],
                "ionization_mode": [
                    "positive",
                    "positive",
                    "positive",
                    "negative",
                    "negative",
                ],
            }
        )
        first = np.array([0, 4])
        second = paired_rows(frame, first)
        np.testing.assert_array_equal(second, [1, 4])
        for a, b in zip(first, second):
            pd.testing.assert_series_equal(
                frame.iloc[a], frame.iloc[b], check_names=False
            )
        np.testing.assert_array_equal(paired_rows(frame, first), second)
