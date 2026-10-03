import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.full_gap_candidate_compatibility import score_function


class FullGapCandidateTests(unittest.TestCase):
    def test_frozen_full_gap_ceiling_is_enforced_and_missing_evidence_is_exact(self):
        weights = np.zeros(60)
        weights[0] = 2.01
        with self.assertRaisesRegex(ValueError, "bonus budget"):
            score_function(weights)
        baseline = np.array([0.1, 0.2], np.float32)
        weights = np.zeros(60)
        weights[24] = 0.25
        feature = np.zeros(60, np.float32)
        feature[24] = 1
        with (
            patch("casmi_ml.direct_models.score_group", return_value=baseline),
            patch(
                "casmi_ml.full_gap_candidate_compatibility.evidence_features",
                return_value=feature,
            ),
        ):
            score = score_function(weights)(
                None,
                None,
                pd.DataFrame([{"row": 1}]),
                None,
                pd.DataFrame({"normalized_smiles": ["c1ccccc1", "CCC"]}),
                None,
            )
        self.assertAlmostEqual(float(score[0]), 0.35, places=6)
        self.assertEqual(score[1].tobytes(), baseline[1].tobytes())
        with patch("casmi_ml.direct_models.score_group", return_value=baseline):
            self.assertIs(
                score_function(weights, True)(
                    None,
                    None,
                    pd.DataFrame(),
                    None,
                    pd.DataFrame({"normalized_smiles": ["CCC", "CCO"]}),
                    None,
                ),
                baseline,
            )

    def test_optimizer_uses_fixed_capacity_and_masked_arm_remains_zero(self):
        scores = np.zeros((12, 16))
        features = np.zeros((12, 16, 60))
        features[:, 0, 24] = 1
        weights, report = fit_bonus(scores, features, 10)
        self.assertGreater(weights[24], 0.05)
        self.assertLessEqual(weights.sum(), 2)
        self.assertEqual(report["maximum_possible_bonus"], 2)
        self.assertLess(report["final_objective"], report["initial_ce"])
        zero, _ = fit_bonus(scores, features, 10, mask=True)
        np.testing.assert_array_equal(zero, np.zeros(60))
