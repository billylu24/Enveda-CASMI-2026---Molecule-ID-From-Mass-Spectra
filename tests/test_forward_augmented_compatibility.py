import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.forward_augmented_compatibility import score_function


class ForwardAugmentedScoreTest(unittest.TestCase):
    def model(self):
        return {
            "weights": np.zeros((2048, 60)),
            "mean_x": np.zeros(2048),
            "mean_y": np.ones(60),
        }

    def test_identity_mask_preserves_original_scores_exactly(self):
        expected = np.asarray([0.11, 0.72], np.float32)
        weights = np.ones(120) / 60
        function = score_function(weights, self.model(), True)
        with patch("casmi_ml.direct_models.score_group", return_value=expected):
            result = function(
                None,
                None,
                None,
                None,
                pd.DataFrame({"normalized_smiles": ["C", "CC"]}),
                np.zeros((2, 2048)),
            )
        np.testing.assert_array_equal(result, expected)

    def test_no_observed_chemistry_cannot_create_support_bonus(self):
        expected = np.asarray([0.11], np.float32)
        function = score_function(np.ones(120) / 60, self.model())
        with (
            patch("casmi_ml.direct_models.score_group", return_value=expected),
            patch(
                "casmi_ml.full_gap_candidate_compatibility.evidence_features",
                return_value=np.zeros(60),
            ),
        ):
            result = function(
                None,
                None,
                pd.DataFrame([{"placeholder": 0}]),
                None,
                pd.DataFrame({"normalized_smiles": ["C"]}),
                np.zeros((1, 2048)),
            )
        np.testing.assert_array_equal(result, expected)

    def test_original_total_bonus_bound_not_doubled_by_extra_channels(self):
        with self.assertRaises(ValueError):
            score_function(np.ones(120) / 30, self.model())
