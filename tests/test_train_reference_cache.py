import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from casmi_ml.reference_reuse import build_shared_reference
from casmi_ml.train_reference_cache import build_cached_reference, prepare, select_rows


class TrainReferenceCacheTest(unittest.TestCase):
    def test_full_train_cache_then_runtime_subset_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.parquet"
            pd.DataFrame(
                [
                    {
                        "molecular_formula": "C2H6O",
                        "normalized_smiles": "CCO",
                        "inchikey14": "A",
                        "ms2_mzs": [45.0, 44.0, 44.01, 10.0],
                        "ms2_normalized_intensities": [0.2, 0.1, 0.3, 0.9],
                        "precursor_error_ppm": 0.0,
                    },
                    {
                        "molecular_formula": "C2H6O",
                        "normalized_smiles": "COC",
                        "inchikey14": "B",
                        "ms2_mzs": [42.0],
                        "ms2_normalized_intensities": [1.0],
                        "precursor_error_ppm": 31.0,
                    },
                    {
                        "molecular_formula": "C3H8O",
                        "normalized_smiles": "CCCO",
                        "inchikey14": "C",
                        "ms2_mzs": [45.0],
                        "ms2_normalized_intensities": [1.0],
                        "precursor_error_ppm": 0.0,
                    },
                ]
            ).to_parquet(train, index=False)
            store = root / "store"
            prepare(train, store)
            neutral = np.array([46.041864812])
            centers = set(neutral) | {60.057514876}
            a, am, ak = build_shared_reference(
                train, neutral, centers, root / "original"
            )
            b, bm, bk = build_cached_reference(
                train, neutral, centers, root / "cached", store=store
            )
            self.assertEqual(a, b)
            self.assertEqual(ak, bk)
            for name in ["data", "indices", "indptr"]:
                np.testing.assert_array_equal(getattr(am, name), getattr(bm, name))
            x, y = [
                sparse.load_npz(root / name / "spectra.npz")
                for name in ["original", "cached"]
            ]
            for name in ["data", "indices", "indptr"]:
                np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
            with (store / "rows.parquet").open("ab") as file:
                file.write(b"changed")
            with self.assertRaises(ValueError):
                build_cached_reference(
                    train, neutral, centers, root / "corrupt", store=store
                )

    def test_mass_filter_retains_original_float_arithmetic(self):
        mass = np.array([100.0, 100.0035, 100.004, np.nan])
        ids = select_rows(mass, [100.0])
        np.testing.assert_array_equal(ids, [0, 1])
        with self.assertRaises(ValueError):
            select_rows(mass, [])
