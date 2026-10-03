import unittest

import numpy as np

from casmi_ml.chemistry import ELECTRON, composition_mass
from casmi_ml.extended_diagnostic_rules import (
    RULES,
    evidence_features,
    structural_support,
)


class ExtendedRulesTests(unittest.TestCase):
    def row(self, mzs, intensities, mode="positive"):
        return {
            "ms2_mzs": mzs,
            "ms2_normalized_intensities": intensities,
            "precursor_mz": 300,
            "instrument_type": "qtof",
            "ionization_mode": mode,
            "adduct": "[M+H]+" if mode == "positive" else "[M-H]-",
        }

    def test_exact_charged_compositions_and_mode_specificity(self):
        self.assertAlmostEqual(RULES[0].mass, composition_mass("C6H5") - ELECTRON)
        self.assertAlmostEqual(RULES[4].mass, composition_mass("PO3") + ELECTRON)
        row = self.row([RULES[0].mass, RULES[4].mass], [1, 1])
        feature = evidence_features(row)
        self.assertEqual(feature[0], 1)
        self.assertEqual(feature[4 * 3], 0)
        row.update(ionization_mode="negative", adduct="[M-H]-")
        feature = evidence_features(row)
        self.assertEqual(feature[0], 0)
        self.assertEqual(feature[4 * 3], 1)
        row["instrument_type"] = "iontrap"
        np.testing.assert_array_equal(evidence_features(row), np.zeros(len(RULES) * 3))

    def test_neutral_loss_endpoints_and_missing_evidence(self):
        methanol = RULES[6].mass
        row = self.row([100, 100 + methanol, 300 - methanol], [0.25, 1, 0.36])
        feature = evidence_features(row)
        self.assertAlmostEqual(float(feature[6 * 3 + 1]), 0.6, places=6)
        self.assertAlmostEqual(float(feature[6 * 3 + 2]), 0.5, places=6)
        row["ms2_mzs"] = []
        row["ms2_normalized_intensities"] = []
        self.assertFalse(evidence_features(row).any())

    def test_actual_candidate_motifs_distinguish_sulfur_phosphorus_and_aromatics(self):
        benzene = structural_support("c1ccccc1")
        alkane = structural_support("CCC")
        phosphate = structural_support("OP(=O)(O)O")
        sulfate = structural_support("COS(=O)(=O)O")
        self.assertTrue(benzene[0])
        self.assertFalse(alkane[0])
        self.assertTrue(phosphate[4])
        self.assertFalse(sulfate[4])
        self.assertTrue(sulfate[8])
        self.assertFalse(phosphate[8])
        self.assertFalse(any(alkane))
