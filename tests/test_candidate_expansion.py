import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

from baseline import PROTON, formula_mass
from casmi_ml.candidate_expansion import evaluate_rows, expanded_pool
from casmi_ml.data import fit_preprocessing
from casmi_ml.models import FingerprintModel
from casmi_ml.secondary_inference import predict


class CandidateExpansionTests(unittest.TestCase):
    def test_offline_extension_adds_isomer_only_below_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mass = formula_mass("C2H6O")
            ethanol = Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))[:14]
            ether = Chem.MolToInchiKey(Chem.MolFromSmiles("COC"))[:14]
            pd.DataFrame(
                [
                    {
                        "inchikey14": ethanol,
                        "normalized_smiles": "CCO",
                        "molecular_formula": "C2H6O",
                        "ms2_mzs": [42.0],
                        "ms2_normalized_intensities": [1.0],
                        "precursor_error_ppm": 0.0,
                    }
                ]
            ).to_parquet(root / "train.parquet")
            frame = pd.DataFrame(
                [
                    {
                        "molecule_id": name,
                        "precursor_mz": mass + PROTON,
                        "adduct": "[M+H]+",
                        "ms2_mzs": peaks,
                        "ms2_normalized_intensities": intensity,
                        "collision_energy_ev": [],
                        "ionization_mode": "positive",
                        "instrument_type": "test",
                    }
                    for name, peaks, intensity in [
                        ("known", [42.0], [1.0]),
                        ("unknown", [], []),
                    ]
                ]
            )
            frame.to_parquet(root / "test.parquet")
            pd.DataFrame(
                [{"inchikey": ethanol, "canonical_smiles": "CCO", "exact_mass": mass}]
            ).to_parquet(root / "coconut.parquet")
            pd.DataFrame(
                [
                    {
                        "inchikey14": ether,
                        "normalized_smiles": "COC",
                        "mass": mass,
                        "origin": "pubchemlite",
                    }
                ]
            ).to_parquet(root / "extra.parquet")
            prep = fit_preprocessing(frame)
            dim = 6 + sum(len(v) + 1 for v in prep["categories"].values())
            model = FingerprintModel("metadata", dim)
            torch.save(
                {
                    "architecture": "metadata",
                    "metadata_dim": dim,
                    "preprocessing": prep,
                    "state_dict": model.state_dict(),
                },
                root / "model.pt",
            )
            recipe = {
                "config": {
                    "kind": "free_top1",
                    "base": "coconut15",
                    "threshold": 0.5,
                    "weight": 0.75,
                },
                "checkpoint": "model.pt",
                "checkpoint_sha256": hashlib.sha256(
                    (root / "model.pt").read_bytes()
                ).hexdigest(),
                "candidate_expansion": {
                    "path": "extra.parquet",
                    "weight": 1.0,
                    "sha256": hashlib.sha256(
                        (root / "extra.parquet").read_bytes()
                    ).hexdigest(),
                },
            }
            (root / "recipe.json").write_text(json.dumps(recipe))
            result = predict(
                root / "recipe.json", root, root / "coconut.parquet", root / "out.csv"
            ).set_index("molecule_id")
            self.assertEqual(result.loc["known", "smiles"], "CCO")
            self.assertEqual(
                set(result.loc["unknown", "smiles"].split(";")), {"CCO", "COC"}
            )

    def test_extension_preserves_existing_structure_and_mass(self):
        original = pd.DataFrame(
            [
                {
                    "inchikey14": "a",
                    "normalized_smiles": "CCO",
                    "mass": 46.0,
                    "origin": "library",
                }
            ]
        )
        external = pd.DataFrame(
            [
                {
                    "inchikey14": "a",
                    "normalized_smiles": "COC",
                    "mass": 47.0,
                    "origin": "pubchemlite",
                },
                {
                    "inchikey14": "b",
                    "normalized_smiles": "CCN",
                    "mass": 45.0,
                    "origin": "pubchemlite",
                },
            ]
        )
        pool = expanded_pool(original, external)
        self.assertEqual(pool.inchikey14.tolist(), ["a", "b"])
        self.assertEqual(pool.iloc[0].to_dict(), original.iloc[0].to_dict())

    def test_new_truth_can_enter_only_expanded_candidate_pool(self):
        row = {
            "key": "b",
            "known": True,
            "confidence": 0.1,
            "old": ["a"],
            "expanded": ["b", "a"],
            "old_pool": ["a"],
            "expanded_pool": ["a", "b"],
        }
        old, _ = evaluate_rows([row], 0, "unknown")
        new, _ = evaluate_rows([row], 1, "unknown")
        self.assertEqual(old["candidate_recall"], 0)
        self.assertEqual(old["mrr25"], 0)
        self.assertEqual(new["candidate_recall"], 1)
        self.assertEqual(new["mrr25"], 1)
        row["confidence"] = 0.9
        guarded, _ = evaluate_rows([row], 1, "known")
        self.assertEqual(guarded["mrr25"], 0)


if __name__ == "__main__":
    unittest.main()
