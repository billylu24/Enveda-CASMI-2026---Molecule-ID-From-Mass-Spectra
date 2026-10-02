import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

from baseline import PROTON, formula_mass
from casmi_ml.data import fit_preprocessing
from casmi_ml.models import FingerprintModel
from casmi_ml.research_models import PeakEncoder
from casmi_ml.secondary_inference import predict


class ResearchInferenceTests(unittest.TestCase):
    def fixture(self, root, family):
        mass = formula_mass("C2H6O")
        a = Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))[:14]
        b = Chem.MolToInchiKey(Chem.MolFromSmiles("COC"))[:14]
        pd.DataFrame(
            [
                {
                    "inchikey14": a,
                    "normalized_smiles": "CCO",
                    "molecular_formula": "C2H6O",
                    "ms2_mzs": [42.0],
                    "ms2_normalized_intensities": [1.0],
                    "precursor_error_ppm": 0.0,
                }
            ]
        ).to_parquet(root / "train.parquet")
        test = pd.DataFrame(
            [
                {
                    "molecule_id": name,
                    "precursor_mz": mass + PROTON,
                    "adduct": "[M+H]+",
                    "ms2_mzs": peaks,
                    "ms2_normalized_intensities": [1.0],
                    "ionization_mode": "positive",
                    "instrument_type": "QTOF",
                    "collision_energy_ev": [],
                }
                for name, peaks in [
                    ("high", [42.0]),
                    ("low", [mass + PROTON - formula_mass("H2O")]),
                ]
            ]
        )
        test.to_parquet(root / "test.parquet")
        pd.DataFrame(
            [
                {"inchikey": a, "canonical_smiles": "CCO", "exact_mass": mass},
                {"inchikey": b, "canonical_smiles": "COC", "exact_mass": mass},
            ]
        ).to_parquet(root / "coconut.parquet")
        prep = fit_preprocessing(test)
        dim = 6 + sum(len(x) + 1 for x in prep["categories"].values())
        model = (
            PeakEncoder(dim)
            if family == "research_peak"
            else FingerprintModel("metadata", dim)
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata_dim": dim,
                "architecture": "metadata",
                "preprocessing": prep,
            },
            root / "model.pt",
        )
        recipe = {
            "encoder_family": family,
            "checkpoint": "model.pt",
            "checkpoint_sha256": hashlib.sha256(
                (root / "model.pt").read_bytes()
            ).hexdigest(),
            "config": {
                "kind": "free_top1",
                "base": "coconut15",
                "threshold": 0.5,
                "weight": 0.75,
            },
        }
        return recipe, a

    def test_real_chemical_inference_protects_high_and_scores_low(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, key = self.fixture(root, "fingerprint")
            recipe["chemistry"] = {
                "rules_version": 1,
                "component": "loss",
                "weight": 1.0,
            }
            (root / "recipe.json").write_text(json.dumps(recipe))
            output = predict(
                root / "recipe.json", root, root / "coconut.parquet", root / "out.csv"
            ).set_index("molecule_id")
            for name in ["high", "low"]:
                first = output.loc[name, "smiles"].split(";")[0]
                self.assertEqual(
                    Chem.MolToInchiKey(Chem.MolFromSmiles(first))[:14], key
                )
            routing = pd.read_csv(str(root / "out.csv") + ".routing.csv").set_index(
                "molecule_id"
            )
            self.assertTrue(routing.loc["high", "protected"])
            self.assertFalse(routing.loc["low", "protected"])

    def test_charge_aware_inference_uses_generic_pool_and_preserves_high_branch(self):
        from unittest.mock import patch

        from casmi_ml.chemistry import parse_adduct
        from casmi_ml.mass_candidates import candidate_window

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, _ = self.fixture(root, "fingerprint")
            recipe["mass_hypothesis"] = "charge_aware_union"
            test = pd.read_parquet(root / "test.parquet")
            ion = parse_adduct("[M+2H]2+")
            test.loc[test.molecule_id == "low", "adduct"] = "[M+2H]2+"
            test.loc[test.molecule_id == "low", "precursor_mz"] = (
                formula_mass("C2H6O") + ion.shift
            ) / 2
            test.to_parquet(root / "test.parquet")
            (root / "recipe.json").write_text(json.dumps(recipe))
            with patch(
                "casmi_ml.mass_candidates.candidate_window", wraps=candidate_window
            ) as window:
                output = predict(
                    root / "recipe.json",
                    root,
                    root / "coconut.parquet",
                    root / "out.csv",
                )
            self.assertEqual(window.call_count, 1)
            self.assertEqual(len(output), 2)
            audit = pd.read_csv(str(root / "out.csv") + ".routing.csv").set_index(
                "molecule_id"
            )
            self.assertTrue(audit.loc["high", "protected"])
            self.assertFalse(audit.loc["low", "protected"])

    def test_peak_checkpoint_uses_true_deployment_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, _ = self.fixture(root, "research_peak")
            (root / "recipe.json").write_text(json.dumps(recipe))
            output = predict(
                root / "recipe.json", root, root / "coconut.parquet", root / "out.csv"
            )
            self.assertEqual(set(output.molecule_id), {"high", "low"})
            self.assertTrue(all(output.smiles.str.len() > 0))


if __name__ == "__main__":
    unittest.main()
