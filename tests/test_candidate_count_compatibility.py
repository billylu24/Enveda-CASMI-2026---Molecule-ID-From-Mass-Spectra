import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.candidate_count_compatibility import score_function


class CountScoreTests(unittest.TestCase):
    def test_zero_mask_and_absent_chemistry_preserve_baseline_exactly(self):
        baseline = np.array([0.1, 0.2], dtype=np.float32)
        candidates = pd.DataFrame(
            {"normalized_smiles": ["c1ccccc1", "c1ccc(-c2ccccc2)cc1"]}
        )
        group = pd.DataFrame([{"row": 1}])
        weights = np.zeros(120)
        weights[84] = 0.2
        with patch(
            "casmi_ml.candidate_count_compatibility.score_group", return_value=baseline
        ):
            for score in [score_function(np.zeros(120)), score_function(weights, True)]:
                self.assertIs(
                    score(None, None, group, None, candidates, None), baseline
                )
            with patch(
                "casmi_ml.candidate_count_compatibility.evidence_features",
                return_value=np.zeros(60),
            ):
                self.assertIs(
                    score_function(weights)(None, None, group, None, candidates, None),
                    baseline,
                )

    def test_extra_motif_counts_only_reward_observed_multiple_matches(self):
        baseline = np.array([0.1, 0.2], dtype=np.float32)
        candidates = pd.DataFrame(
            {"normalized_smiles": ["c1ccccc1", "c1ccc(-c2ccccc2)cc1"]}
        )
        weights = np.zeros(120)
        weights[84] = 0.3
        observed = np.zeros(60, dtype=np.float32)
        observed[24] = 1
        with (
            patch(
                "casmi_ml.candidate_count_compatibility.score_group",
                return_value=baseline,
            ),
            patch(
                "casmi_ml.candidate_count_compatibility.evidence_features",
                return_value=observed,
            ),
        ):
            actual = score_function(weights)(
                None, None, pd.DataFrame([{"row": 1}]), None, candidates, None
            )
        self.assertEqual(actual[0].tobytes(), baseline[0].tobytes())
        self.assertAlmostEqual(actual[1], 0.3, places=6)
        with self.assertRaisesRegex(ValueError, "fixed2"):
            score_function(np.ones(120))


if __name__ == "__main__":
    unittest.main()
