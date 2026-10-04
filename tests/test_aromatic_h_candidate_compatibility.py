import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.aromatic_h_candidate_compatibility import score_function


class AromaticHScoreTests(unittest.TestCase):
    def test_zero_mask_and_absent_evidence_preserve_baseline_exactly(self):
        baseline = np.array([0.1, 0.2], dtype=np.float32)
        candidates = pd.DataFrame({"normalized_smiles": ["c1ccccc1", "CCc1ccccc1"]})
        group = pd.DataFrame([{"row": 1}])
        weights = np.zeros(60)
        weights[27] = 0.2
        with patch(
            "casmi_ml.aromatic_h_candidate_compatibility.score_group",
            return_value=baseline,
        ):
            for fn in (score_function(np.zeros(60)), score_function(weights, True)):
                self.assertIs(fn(None, None, group, None, candidates, None), baseline)
            with patch(
                "casmi_ml.aromatic_h_candidate_compatibility.evidence_features",
                return_value=np.zeros(60),
            ):
                self.assertIs(
                    score_function(weights)(None, None, group, None, candidates, None),
                    baseline,
                )

    def test_refines_detachable_ion_and_retains_loss_channels(self):
        baseline = np.array([0.1, 0.2], np.float32)
        candidates = pd.DataFrame({"normalized_smiles": ["Cc1ccccc1", "CCc1ccccc1"]})
        observed = np.zeros(60, np.float32)
        observed[27] = 1
        weights = np.zeros(60)
        weights[27] = 0.3
        with (
            patch(
                "casmi_ml.aromatic_h_candidate_compatibility.score_group",
                return_value=baseline,
            ),
            patch(
                "casmi_ml.aromatic_h_candidate_compatibility.evidence_features",
                return_value=observed,
            ),
        ):
            actual = score_function(weights)(
                None, None, pd.DataFrame([{"row": 1}]), None, candidates, None
            )
        self.assertEqual(actual[0].tobytes(), baseline[0].tobytes())
        self.assertAlmostEqual(actual[1], 0.5, places=6)
        with self.assertRaises(ValueError):
            score_function(np.ones(60))


if __name__ == "__main__":
    unittest.main()
