import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.candidate_diagnostic_compatibility import (
    compatibility_features,
    score_function,
    structural_support,
)


class CandidateDiagnosticCompatibilityTests(unittest.TestCase):
    def test_same_query_compares_actual_candidate_motifs_and_absence_is_zero(self):
        chemical = np.zeros(24, np.float32)
        chemical[1] = 0.6
        support = np.asarray(
            [structural_support("CCO"), structural_support("CCC")], np.float32
        )
        feature = compatibility_features(chemical, support)
        self.assertEqual(feature.shape, (2, 24))
        self.assertAlmostEqual(float(feature[0, 1]), 0.6, places=6)
        self.assertEqual(float(feature[1].sum()), 0)
        np.testing.assert_array_equal(
            compatibility_features(np.zeros(24, np.float32), support),
            np.zeros((2, 24), np.float32),
        )
        with self.assertRaisesRegex(ValueError, "Chemical24"):
            compatibility_features(np.zeros(23), support)

    def test_zero_and_masked_paths_return_original_score_array_exactly(self):
        baseline = np.asarray([0.1, -0.2, 0.3], np.float32)
        for weights, mask in [(np.zeros(24), False), (np.ones(24) * 0.001, True)]:
            with patch("casmi_ml.direct_models.score_group", return_value=baseline):
                fn = score_function(weights, mask)
                actual = fn(
                    None,
                    None,
                    pd.DataFrame(),
                    None,
                    pd.DataFrame({"normalized_smiles": ["CCO", "CCC", "CCN"]}),
                    None,
                )
                self.assertIs(actual, baseline)
        with self.assertRaisesRegex(ValueError, "Nonnegative"):
            score_function(-np.ones(24))

    def test_bonus_cap_and_candidate_differences_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "bonus budget"):
            score_function(np.ones(24))
        weights = np.zeros(24, np.float32)
        weights[1] = 0.04
        chemical = np.zeros(24, np.float32)
        chemical[1] = 1
        candidates = pd.DataFrame({"normalized_smiles": ["CCO", "CCC"]})
        baseline = np.asarray([0.1, -0.2], np.float32)
        with (
            patch("casmi_ml.direct_models.score_group", return_value=baseline),
            patch(
                "casmi_ml.candidate_diagnostic_compatibility.evidence_features",
                return_value=chemical,
            ),
        ):
            actual = score_function(weights)(
                None, None, pd.DataFrame([{"row_id": 1}]), None, candidates, None
            )
        self.assertAlmostEqual(float(actual[0]), 0.14, places=6)
        self.assertEqual(actual[1].tobytes(), baseline[1].tobytes())
