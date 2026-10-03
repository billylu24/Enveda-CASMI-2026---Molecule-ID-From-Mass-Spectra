"""Protect label-free inference and exact full-ranking handoffs."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_routed_inference import extend


class RoutedInferenceBoundaryTests(unittest.TestCase):
    def test_truth_columns_are_rejected_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for forbidden in (
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "fingerprint",
            ):
                path = root / "test.parquet"
                pd.DataFrame(
                    {"molecule_id": ["opaque1"], forbidden: ["label"]}
                ).to_parquet(path)
                with self.assertRaisesRegex(ValueError, "unlabeled input"):
                    extend(
                        path,
                        root / "missing.csv",
                        None,
                        None,
                        root / "output.csv",
                        root,
                        {},
                    )

    def test_current_full_ranking_must_match_baseline_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"molecule_id": ["opaque1"]}).to_parquet(root / "test.parquet")
            pd.DataFrame({"molecule_id": ["opaque1"], "smiles": ["CCO"]}).to_csv(
                root / "base.csv", index=False
            )
            (root / "full.json").write_text(
                json.dumps([{"molecule_id": "opaque1", "smiles": ["COC"]}])
            )
            with self.assertRaisesRegex(ValueError, "frozen baseline CSV"):
                extend(
                    root / "test.parquet",
                    root / "base.csv",
                    root / "full.json",
                    None,
                    root / "output.csv",
                    root,
                    {},
                )
