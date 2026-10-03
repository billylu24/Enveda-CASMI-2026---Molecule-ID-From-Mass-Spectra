import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import formula_mass, load_candidates, vectorize
from casmi_ml.observed_reference_keys import load_observed_keys, usable_spectrum


class ObservedReferenceKeyTests(unittest.TestCase):
    def test_exact_nonempty_vector_criterion_at_all_peak_boundaries(self):
        cases = [
            ([], []),
            ([39.999, 1250], [1.0, 1.0]),
            ([40.0, 1249.999], [0.01, 0.01]),
            ([100.0], [0.009999]),
            ([np.nan, np.inf, 100.0], [1.0, 1.0, np.nan]),
            ([100.0] * 101, [0.1] * 101),
        ]
        for mzs, intensity in cases:
            self.assertEqual(
                usable_spectrum(mzs, intensity), bool(vectorize(mzs, intensity))
            )

    def test_membership_matches_original_mass_quality_and_nonempty_filters(self):
        mass = formula_mass("C2H6O")
        rows = []
        for i, (smiles, key, mz, intensity, ppm, formula) in enumerate(
            [
                ("CCO", "valid", [40.0], [0.01], None, "C2H6O"),
                ("CCO", "valid", [100.0], [0.5], 0.0, "C2H6O"),
                ("CCO", "invalid", [40.0], [0.009], 0.0, "C2H6O"),
                ("CCO", "invalid", [39.99], [0.5], 0.0, "C2H6O"),
                ("CCO", "invalid", [1250.0], [0.5], 0.0, "C2H6O"),
                ("CCO", "ppm_boundary", [40.0], [0.5], -30.0, "C2H6O"),
                ("CCO", "invalid", [40.0], [0.5], 30.01, "C2H6O"),
                ("CCO", "nan_ppm", [40.0], [0.5], np.nan, "C2H6O"),
                ("CCO", "invalid", [], [], 0.0, "C2H6O"),
                ("", "invalid", [40.0], [0.5], 0.0, "C2H6O"),
                ("CCO", "", [40.0], [0.5], 0.0, "C2H6O"),
                ("CCO", "invalid", [40.0], [0.5], 0.0, "C3H8O"),
                ("CCO", "invalid", [40.0], [0.5], 0.0, "bad"),
            ]
        ):
            rows.append(
                {
                    "molecular_formula": formula,
                    "normalized_smiles": smiles,
                    "inchikey14": key,
                    "ms2_mzs": mz,
                    "ms2_normalized_intensities": intensity,
                    "precursor_error_ppm": ppm,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            for target in (mass, mass / (1 + 34.99e-6), mass / (1 + 35.01e-6)):
                original, _ = load_candidates(path, [target])
                self.assertEqual(
                    load_observed_keys(path, [target]), {r[1] for r in original}
                )
