import unittest

import numpy as np

from casmi_ml.chemistry import parse_adduct
from casmi_ml.data import features, fit_preprocessing
from casmi_ml.dimer_input_view import input_view


class ObservableDimerInputTests(unittest.TestCase):
    def row(self):
        adduct = "[2M+Na]+"
        mass = 200.0
        mono = parse_adduct("[M+Na]+")
        ion = parse_adduct(adduct)
        return {
            "adduct": adduct,
            "precursor_mz": 2 * mass + ion.shift,
            "ms2_mzs": [100.0, mass + mono.shift, 300.0],
            "ms2_normalized_intensities": [0.5, 1.0, 0.8],
            "instrument_type": "Orbitrap",
            "ionization_mode": "positive",
            "collision_energy_ev": [20.0, 40.0],
        }

    def test_measured_monomer_view_keeps_acquisition_metadata(self):
        row = self.row()
        result, changed = input_view(row)
        self.assertTrue(changed)
        self.assertEqual(result["adduct"], "[M+Na]+")
        self.assertEqual(result["ms2_mzs"], [100.0])
        self.assertEqual(result["collision_energy_ev"], row["collision_energy_ev"])
        self.assertEqual(result["ionization_mode"], row["ionization_mode"])

    def test_absent_marker_leaves_exact_original_input(self):
        row = self.row()
        row["ms2_normalized_intensities"][1] = 0.01
        result, changed = input_view(row)
        self.assertFalse(changed)
        self.assertIs(result, row)

    def test_training_view_matches_flagged_runtime_features(self):
        import pandas as pd

        row = self.row()
        config = fit_preprocessing(pd.DataFrame([row]))
        transformed, _ = input_view(row)
        expected = features(transformed, config)
        actual = features(row, dict(config, dimer_product_input=True))
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)

    def test_unflagged_checkpoint_keeps_legacy_features(self):
        import pandas as pd

        row = self.row()
        config = fit_preprocessing(pd.DataFrame([row]))
        expected = features(row, config)
        actual = features(row, dict(config, dimer_product_input=False))
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
