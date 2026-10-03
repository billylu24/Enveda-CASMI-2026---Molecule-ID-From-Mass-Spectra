import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.max_candidate_compatibility import score_function


class MaxCandidateTests(unittest.TestCase):
    def test_single_strong_spectrum_is_not_diluted_and_duplicate_vote_has_no_bonus(
        self,
    ):
        baseline = np.array([0.1, 0.2], np.float32)
        candidates = pd.DataFrame({"normalized_smiles": ["COC", "CCC"]})
        weights = np.zeros(60)
        weights[24 + 6 * 3 + 1] = 0.04

        def features(row, precise):
            result = np.zeros(60, np.float32)
            result[24 + 6 * 3 + 1] = row["strength"]
            return result

        with (
            patch("casmi_ml.direct_models.score_group", return_value=baseline),
            patch(
                "casmi_ml.max_candidate_compatibility.evidence_features",
                side_effect=features,
            ),
        ):
            fn = score_function(weights)
            rows = pd.DataFrame({"strength": [0.6, 0]})
            scores = fn(None, None, rows, None, candidates, None)
            duplicated = fn(None, None, pd.concat([rows, rows]), None, candidates, None)
            np.testing.assert_array_equal(scores, duplicated)
            self.assertAlmostEqual(float(scores[0]), 0.124, places=6)
            self.assertEqual(scores[1].tobytes(), baseline[1].tobytes())
            absent = fn(
                None, None, pd.DataFrame({"strength": [0, 0]}), None, candidates, None
            )
            self.assertIs(absent, baseline)
