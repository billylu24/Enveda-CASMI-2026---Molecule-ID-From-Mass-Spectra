import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.bounded_candidate_bonus import fit_bonus
from casmi_ml.candidate_compatibility_training import fit_bonus as original_fit
from casmi_ml.extended_candidate_compatibility import (
    DIM,
    compatibility_features,
    evidence_features,
    score_function,
    structural_support,
)
from casmi_ml.extended_diagnostic_rules import RULES


class ExtendedCandidateTests(unittest.TestCase):
    def test_extended_ions_interact_with_actual_candidate_structures(self):
        row = {
            "ms2_mzs": [RULES[0].mass],
            "ms2_normalized_intensities": [1],
            "precursor_mz": 300,
            "instrument_type": "qtof",
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
        }
        chemical = evidence_features(row)
        self.assertEqual(len(chemical), 60)
        self.assertEqual(chemical[24], 1)
        support = np.array([structural_support("c1ccccc1"), structural_support("CCC")])
        feature = compatibility_features(chemical, support)
        self.assertEqual(feature.shape, (2, 60))
        self.assertEqual(feature[0, 24], 1)
        self.assertEqual(feature[1, 24], 0)
        baseline = np.array([0.1, 0.2], np.float32)
        weights = np.zeros(DIM)
        weights[24] = 0.04
        with patch("casmi_ml.direct_models.score_group", return_value=baseline):
            score = score_function(weights)(
                None,
                None,
                pd.DataFrame([row]),
                None,
                pd.DataFrame({"normalized_smiles": ["c1ccccc1", "CCC"]}),
                None,
            )
        self.assertAlmostEqual(float(score[0]), 0.14, places=6)
        self.assertEqual(score[1].tobytes(), baseline[1].tobytes())

    def test_general_optimizer_preserves_original24_solution_and_handles60(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(-0.3, 0.3, (20, 16))
        interactions = rng.uniform(0, 1, (20, 16, 24))
        old, old_report = original_fit(scores, interactions, 10)
        new, new_report = fit_bonus(scores, interactions, 10)
        np.testing.assert_array_equal(old, new)
        self.assertEqual(old_report, new_report)
        features = np.zeros((20, 16, 60))
        features[:, 0, 48] = 1
        weights, report = fit_bonus(scores, features, 10)
        self.assertGreater(weights[48], 0.049)
        self.assertLessEqual(weights.sum(), 0.05)
        self.assertLess(report["final_objective"], report["initial_ce"])
        mask, _ = fit_bonus(scores, features, 10, mask=True)
        np.testing.assert_array_equal(mask, np.zeros(60))

    def test_zero_and_mask_preserve_original_score_without_chemical_work(self):
        baseline = np.array([0.1], np.float32)
        for weights, mask in [(np.zeros(60), False), (np.ones(60) * 0.0001, True)]:
            with (
                patch("casmi_ml.direct_models.score_group", return_value=baseline),
                patch(
                    "casmi_ml.extended_candidate_compatibility.evidence_features",
                    side_effect=AssertionError("must bypass"),
                ),
            ):
                actual = score_function(weights, mask)(
                    None,
                    None,
                    pd.DataFrame(),
                    None,
                    pd.DataFrame({"normalized_smiles": ["CCC"]}),
                    None,
                )
                self.assertIs(actual, baseline)
