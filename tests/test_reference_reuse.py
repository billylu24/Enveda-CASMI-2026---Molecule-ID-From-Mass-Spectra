import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from baseline import formula_mass, load_candidates, make_matrix
from casmi_ml.reference_reuse import build_shared_reference


class SharedReferenceTests(unittest.TestCase):
    def test_real_loader_subset_and_union_csr_preserve_every_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for formula, key, peaks, ppm in [
                ("C2H6O", "a", [42, 43], 0),
                ("C3H8O", "b", [44], 0),
                ("C2H6O", "c", [45], 31),
                ("C2H6O", "d", [39], 0),
                ("C2H6O", "e", [42], 0),
            ]:
                rows.append(
                    {
                        "molecular_formula": formula,
                        "normalized_smiles": "CCO",
                        "inchikey14": key,
                        "ms2_mzs": peaks,
                        "ms2_normalized_intensities": [1.0] * len(peaks),
                        "precursor_error_ppm": ppm,
                    }
                )
            pd.DataFrame(rows).to_parquet(root / "train.parquet")
            neutral = [formula_mass("C2H6O")]
            centers = neutral + [formula_mass("C3H8O")]
            legacy, _ = load_candidates(root / "train.parquet", np.array(neutral))
            union, _ = load_candidates(root / "train.parquet", np.array(centers))
            actual, matrix, keys = build_shared_reference(
                root / "train.parquet", neutral, centers, root / "shared"
            )
            self.assertEqual(actual, [r[:3] for r in legacy])
            self.assertEqual(keys, {r[1] for r in union})
            for expected, found in [
                (make_matrix([r[3] for r in legacy]), matrix),
                (
                    make_matrix([r[3] for r in union]),
                    sparse.load_npz(root / "shared/spectra.npz"),
                ),
            ]:
                self.assertEqual(expected.shape, found.shape)
                for field in ("data", "indices", "indptr"):
                    np.testing.assert_array_equal(
                        getattr(expected, field), getattr(found, field)
                    )

    def test_secondary_inference_uses_one_scan_with_identical_full_rankings(self):
        import json
        from unittest.mock import patch

        from casmi_ml.secondary_inference import predict
        from tests.test_research_inference import ResearchInferenceTests

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, _ = ResearchInferenceTests().fixture(root, "fingerprint")
            recipe.update(
                mass_hypothesis="charge_aware_union",
                generation={"adaptive_prefix": "second_unreferenced"},
                external_routed={
                    "high_fragment": {
                        "strong_slots": {
                            "reference_threshold": 0.5,
                            "original_prefix": 3,
                            "slots": 3,
                        }
                    }
                },
            )
            (root / "recipe.json").write_text(json.dumps(recipe))
            with patch(
                "casmi_ml.secondary_inference.load_candidates", wraps=load_candidates
            ) as scan:
                predict(
                    root / "recipe.json",
                    root,
                    root / "coconut.parquet",
                    root / "legacy.csv",
                    full_rankings=root / "legacy.json",
                )
                self.assertEqual(scan.call_count, 3)
            recipe["reference_runtime"] = {"scan": "shared_union_v1"}
            (root / "recipe.json").write_text(json.dumps(recipe))
            with patch(
                "casmi_ml.reference_reuse.load_candidates", wraps=load_candidates
            ) as scan:
                predict(
                    root / "recipe.json",
                    root,
                    root / "coconut.parquet",
                    root / "shared.csv",
                    full_rankings=root / "shared.json",
                )
                self.assertEqual(scan.call_count, 1)
            self.assertEqual(
                json.loads((root / "legacy.json").read_text()),
                json.loads((root / "shared.json").read_text()),
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(root / "legacy.csv"), pd.read_csv(root / "shared.csv")
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(str(root / "legacy.csv") + ".routing.csv"),
                pd.read_csv(str(root / "shared.csv") + ".routing.csv"),
            )
