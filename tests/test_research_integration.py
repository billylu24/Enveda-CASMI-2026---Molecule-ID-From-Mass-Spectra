import json
import tempfile
import unittest
from pathlib import Path

from casmi_ml.chemistry_experiment import chemical_ranking
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_evaluation import select_representations
from casmi_ml.research_protocol import freeze


class ResearchIntegrationTests(unittest.TestCase):
    def test_chemical_protection_and_rank_evidence(self):
        row = {
            "base": ["a", "b", "c"],
            "confidence": 0.9,
            "chemistry": {"combined": {"a": 0.0, "b": 0.0, "c": 5.0}},
            "fragment_scores": {"a": 0.0, "b": 0.0, "c": 10.0},
        }
        self.assertEqual(chemical_ranking(row, "combined", 0.5), row["base"])
        row["confidence"] = 0.1
        self.assertEqual(chemical_ranking(row, "combined", 1.0)[0], "c")
        self.assertEqual(
            set(chemical_ranking(row, "combined_fragment", 0.25)), {"a", "b", "c"}
        )
        row["fragment_scores"] = {}
        self.assertEqual(chemical_ranking(row, "fragment", 0.5), row["base"])

    def test_frozen_protocol_normalizes_json_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            freeze(path, {"adducts": ("a", "b")})
            freeze(path, {"adducts": ("a", "b")})
            self.assertEqual(json.loads(path.read_text()), {"adducts": ["a", "b"]})

    def test_budget_persists_and_does_not_reset_on_new_run(self):
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                "os.environ", {"CASMI_GPU_LOCK": str(Path(directory) / "gpu.lock")}
            ),
        ):
            budget = StageBudget(directory, "dino", "first", 8, limit=10)
            budget.started -= 6
            budget.close()
            second = StageBudget(directory, "dino", "second", 8, limit=10)
            self.assertAlmostEqual(second.allowance, 4, delta=0.02)
            second.started -= 4
            second.close()
            with self.assertRaises(RuntimeError):
                StageBudget(directory, "dino", "third", 8, limit=10)

    def test_no_development_winner_never_opens_holdout(self):
        from casmi_ml.chemistry_experiment import accept
        from casmi_ml.metfrag import digest
        from casmi_ml.research_evaluation import representation_accept

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze(root / "protocol.json", {"version": 1})
            freeze(
                root / "chemical_selection.json",
                {
                    "accepted_for_holdout": False,
                    "protocol_sha256": digest(root / "protocol.json"),
                },
            )
            freeze(
                root / "representation_selection.json", {"accepted_for_holdout": False}
            )
            self.assertFalse(accept(root)["holdout_opened"])
            self.assertFalse(representation_accept(root)["holdout_opened"])

    def test_representation_selection_requires_all_controls(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            select_representations(directory)


if __name__ == "__main__":
    unittest.main()


class GenerationFallbackTests(unittest.TestCase):
    def test_unlabeled_generation_protects_retrieval_without_query_formula(self):
        from unittest.mock import patch

        import pandas as pd
        import torch

        from casmi_ml.generation_experiment import FormulaPredictor
        from casmi_ml.generation_inference import predict
        from casmi_ml.metfrag import digest
        from casmi_ml.research_models import SmilesDecoder, SmilesVocabulary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "encoder.pt"
            encoder.write_bytes(b"fake_encoder_for_protected_branch")
            v = SmilesVocabulary.fit(["CCO"])
            decoder = SmilesDecoder(len(v.tokens), 18)
            formula = FormulaPredictor(4)
            checkpoint = root / "generator.pt"
            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "formula": formula.state_dict(),
                    "condition_dim": 4,
                    "vocabulary": v.tokens,
                    "config": {"encoder_sha256": digest(encoder)},
                },
                checkpoint,
            )
            pd.DataFrame(
                [{"molecule_id": "a", "ms2_mzs": [], "ms2_normalized_intensities": []}]
            ).to_parquet(root / "test.parquet")
            pd.DataFrame([{"molecule_id": "a", "smiles": "CCO"}]).to_csv(
                root / "baseline.csv", index=False
            )
            pd.DataFrame([{"molecule_id": "a", "protected": True}]).to_csv(
                str(root / "baseline.csv") + ".routing.csv", index=False
            )
            with (
                patch("casmi_ml.generation_inference.ENCODER", encoder),
                patch(
                    "casmi_ml.generation_inference.load_deployment_checkpoint",
                    return_value=(torch.nn.Identity(), {}),
                ),
            ):
                result = predict(
                    checkpoint,
                    root / "test.parquet",
                    root / "baseline.csv",
                    root / "out.csv",
                )
            self.assertEqual(result.smiles.tolist(), ["CCO"])
            report = json.loads(
                Path(str(root / "out.csv") + ".report.json").read_text()
            )
            self.assertFalse(report["formula_oracle_used"])


class GenerationReportTests(unittest.TestCase):
    def test_unknown_pairing_does_not_use_known_scenario_rows(self):
        from unittest.mock import patch

        import pandas as pd

        from casmi_ml.research_evaluation import generated_report

        frame = pd.DataFrame(
            [
                {
                    "inchikey14": "truth",
                    "molecular_formula": "C2H6O",
                    "normalized_smiles": "CCO",
                }
            ]
        )
        unknown = [
            {
                "key": "truth",
                "confidence": 0.1,
                "base": ["wrong", "truth"],
                "available": {"union35": ["wrong", "truth"], "coconut15": []},
            }
        ]
        known = [
            {**unknown[0], "known": True, "confidence": 0.9, "base": ["truth", "wrong"]}
        ]
        generated = [
            {
                "key": "truth",
                "candidates": [{"key": "truth", "smiles": "CCO"}],
                "statistics": {
                    "samples": 1,
                    "terminated": 1,
                    "valid": 1,
                    "mass_matching": 1,
                    "unique_mass_matching": 1,
                },
                "formula_hypotheses": [{"formula": "C2H6O"}],
            }
        ]
        with (
            patch("pandas.read_parquet", return_value=frame),
            patch(
                "casmi_ml.chemistry_experiment.reference_records",
                side_effect=[unknown, known],
            ),
            patch(
                "casmi_ml.research_evaluation.ablation.route",
                side_effect=lambda r, _: r["base"],
            ),
        ):
            result = generated_report(Path("unused"), generated)
        self.assertEqual(result["known"]["paired"]["difference"], 0.0)
        self.assertEqual(
            result["merged_paired"]["difference"],
            result["merged"]["mrr25"] - result["baseline"]["mrr25"],
        )
        self.assertEqual(result["merged_paired"]["difference"], 0.5)


class AtomicReportTests(unittest.TestCase):
    def test_concurrent_writes_leave_one_complete_json_and_no_temp_files(self):
        from concurrent.futures import ThreadPoolExecutor

        from casmi_ml.data import write_json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            values = [{"run": i, "rows": list(range(100))} for i in range(40)]
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(lambda value: write_json(path, value), values))
            self.assertIn(json.loads(path.read_text()), values)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class ReleaseGateTests(unittest.TestCase):
    def test_failed_acceptance_never_creates_release_directory(self):
        from casmi_ml.research_release import prepare_release

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chemical_acceptance.json").write_text(
                json.dumps({"accepted": False})
            )
            with self.assertRaisesRegex(ValueError, "did not pass"):
                prepare_release(root)
            self.assertFalse((root / "release").exists())

    def test_changed_selection_never_creates_release_directory(self):
        from casmi_ml.research_release import prepare_release

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chemical_acceptance.json").write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "selection_sha256": "different-frozen-selection",
                    }
                )
            )
            (root / "chemical_selection.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed after acceptance"):
                prepare_release(root)
            self.assertFalse((root / "release").exists())
