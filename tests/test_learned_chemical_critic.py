import unittest

import numpy as np

from casmi_ml.learned_chemical_critic import (
    NAMES,
    fit_weights,
    observed_strength,
    structural_support,
)


class LearnedChemicalCriticTests(unittest.TestCase):
    def test_conditional_fit_learns_support_and_unsupported_features_stay_zero(self):
        differences = np.zeros((10, len(NAMES)))
        differences[:, 0] = 0.5
        weights, objective = fit_weights(differences)
        self.assertGreater(weights[0], 0)
        np.testing.assert_array_equal(weights[1:], np.zeros(len(NAMES) - 1))
        self.assertLess(objective, np.log(2))
        np.testing.assert_array_equal(weights, fit_weights(differences)[0])
        with self.assertRaises(ValueError):
            fit_weights(np.empty((0, len(NAMES))))

    def test_group_evidence_uses_only_observation_and_duplicate_max(self):
        row = {
            "precursor_mz": 100.0,
            "adduct": "[M+H]+",
            "instrument_type": "Orbitrap",
            "ionization_mode": "positive",
            "ms2_mzs": [81.989435316],
            "ms2_normalized_intensities": [1.0],
        }
        strength = observed_strength([row])
        self.assertGreater(strength[0], 0)
        np.testing.assert_array_equal(strength, observed_strength([row, row]))
        np.testing.assert_array_equal(
            strength,
            observed_strength(
                [
                    {
                        **row,
                        "normalized_smiles": "wrong",
                        "molecular_formula": "wrong",
                        "inchikey14": "wrong",
                    }
                ]
            ),
        )
        self.assertGreater(structural_support("CCO")[0], structural_support("CCC")[0])
