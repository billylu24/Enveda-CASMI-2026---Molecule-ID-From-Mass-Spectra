import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.energy_conditioned_compatibility import score_function


class EnergyConditionedCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.vocab = {"adduct": ["[M+H]+"], "instrument_type": ["QTOF"]}
        self.forward = {
            "weights": np.zeros((4101, 60)),
            "mean_x": np.zeros(4101),
            "mean_y": np.ones(60) * 0.5,
        }
        self.candidates = pd.DataFrame({"normalized_smiles": ["CCO", "CCC"]})
        self.fps = np.zeros((2, 2048))
        self.baseline = np.array([0.1, 0.2], np.float32)

    def evaluate(self, group, weights, mask=False, evidence=None):
        with (
            patch("casmi_ml.direct_models.score_group", return_value=self.baseline),
            patch(
                "casmi_ml.full_gap_candidate_compatibility.evidence_features",
                return_value=np.ones(60) if evidence is None else evidence,
            ),
        ):
            return score_function(weights, self.forward, self.vocab, mask)(
                None, None, group, None, self.candidates, self.fps
            )

    def test_zero_mask_and_no_observation_preserve_original_scores_exact(self):
        group = pd.DataFrame(
            {
                "collision_energy_ev": [[20]],
                "adduct": ["[M+H]+"],
                "instrument_type": ["QTOF"],
            }
        )
        weights = np.zeros(120)
        weights[60] = 1
        np.testing.assert_array_equal(
            self.evaluate(group, weights, True), self.baseline
        )
        np.testing.assert_array_equal(
            self.evaluate(group, np.zeros(120)), self.baseline
        )
        np.testing.assert_array_equal(
            self.evaluate(group, weights, evidence=np.zeros(60)), self.baseline
        )

    def test_missing_merged_energy_zero_extra_and_mean_is_per_original_row(self):
        weights = np.zeros(120)
        weights[60] = 1
        for energy in (None, [20, 40]):
            group = pd.DataFrame({"collision_energy_ev": [energy]})
            np.testing.assert_array_equal(self.evaluate(group, weights), self.baseline)
        group = pd.DataFrame(
            {
                "collision_energy_ev": [[20], None],
                "adduct": ["[M+H]+"] * 2,
                "instrument_type": ["QTOF"] * 2,
            }
        )
        np.testing.assert_allclose(self.evaluate(group, weights), self.baseline + 0.25)

    def test_total_bonus_ceiling_is_not_doubled_and_invalid_forward_rejected(self):
        with self.assertRaises(ValueError):
            score_function(np.ones(120), self.forward, self.vocab)
        with self.assertRaises(ValueError):
            score_function(
                np.zeros(120),
                {
                    "weights": np.zeros((4100, 60)),
                    "mean_x": np.zeros(4101),
                    "mean_y": np.zeros(60),
                },
                self.vocab,
            )
