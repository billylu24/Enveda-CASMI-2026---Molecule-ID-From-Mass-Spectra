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


class ProtectedGenerationRoutingTests(unittest.TestCase):
    def test_confidence_accepts_rounding_only(self):
        from casmi_ml.generation_inference import validated_confidence

        self.assertEqual(
            validated_confidence({"high": 1.0000002, "low": -0.0000002, "mid": 0.1}),
            {"high": 1.0, "low": 0.0, "mid": 0.1},
        )
        for invalid in [float("nan"), float("inf"), -0.01, 1.01]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validated_confidence({"query": invalid})

    def test_open_protected_preserves_prefix_honors_branch_and_timeout(self):
        from unittest.mock import MagicMock, patch

        import numpy as np
        import pandas as pd
        import torch
        from rdkit import Chem

        from casmi_ml.generation_inference import predict
        from casmi_ml.metfrag import digest

        base = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
        for allowed, timeout, frequency_weight, adaptive, expanded, gate_confidence in [
            (True, False, 0.0, None, None, None),
            (True, False, 1.0, None, None, None),
            (False, False, 1.0, None, None, None),
            (True, True, 1.0, None, None, None),
            (True, False, 1.0, "second_unreferenced", None, None),
            (False, False, 1.0, "second_unreferenced", 2, None),
            (False, True, 1.0, "second_unreferenced", 2, None),
            (True, False, 1.0, "second_unreferenced", 2, None),
            (True, False, 1.0, "second_unreferenced", 2, 0.1),
            (True, False, 1.0, "second_unreferenced", 2, 0.7),
            (True, False, 1.0, "second_unreferenced", 2, 1.0000002),
            (False, False, 1.0, "second_unreferenced", 2, 0.1),
            (True, True, 1.0, "second_unreferenced", 2, 0.1),
        ]:
            with (
                self.subTest(
                    allowed=allowed, timeout=timeout, frequency_weight=frequency_weight
                ),
                tempfile.TemporaryDirectory() as d,
            ):
                root = Path(d)
                encoder = root / "encoder.pt"
                encoder.write_bytes(b"encoder")
                (root / "model.pt").write_bytes(b"generator")
                (root / "critic.pt").write_bytes(b"critic")
                pd.DataFrame(
                    [
                        {
                            "molecule_id": "query",
                            "ms2_mzs": [],
                            "ms2_normalized_intensities": [],
                        }
                    ]
                ).to_parquet(root / "test.parquet")
                pd.DataFrame(
                    [{"molecule_id": "query", "smiles": ";".join(base)}]
                ).to_csv(root / "base.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "molecule_id": "query",
                            "protected": True,
                            "generation_allowed": allowed,
                            "second_candidate_has_reference": False,
                            "confidence": gate_confidence
                            if gate_confidence is not None
                            else 0.7,
                        }
                    ]
                ).to_csv(root / "route.csv", index=False)
                (root / "full.json").write_text(
                    json.dumps([{"molecule_id": "query", "smiles": base}])
                )
                decoder, formula = MagicMock(), MagicMock()
                formula.soft.return_value = torch.zeros(1, 1)
                decoder.generate.return_value = (
                    torch.zeros(1, 1),
                    torch.zeros(1),
                    torch.ones(1),
                )
                if timeout:
                    decoder.generate.side_effect = TimeoutError("deadline")
                with (
                    patch(
                        "torch.load",
                        return_value={
                            "config": {"encoder_sha256": digest(encoder)},
                            "encoder_sha256": digest(encoder),
                            "architecture": "fingerprint",
                            "state_dict": {},
                        },
                    ),
                    patch(
                        "casmi_ml.generation_inference.load_model",
                        return_value=(decoder, formula, None),
                    ),
                    patch(
                        "casmi_ml.generation_inference.load_deployment_checkpoint",
                        return_value=(torch.nn.Identity(), {"preprocessing": {}}),
                    ),
                    patch(
                        "casmi_ml.generation_sampling.condition_for_group",
                        return_value=torch.zeros(1, 1),
                    ),
                    patch(
                        "casmi_ml.generation_sampling.sampling_seed", return_value=42
                    ),
                    patch("casmi_ml.direct_models.DirectRanker"),
                    patch(
                        "casmi_ml.direct_models.score_group",
                        return_value=np.array([0.0, 0.4]),
                    ),
                    patch(
                        "casmi_ml.generation_inference.validate_generated",
                        return_value=(
                            [
                                {
                                    "key": Chem.MolToInchiKey(Chem.MolFromSmiles("N"))[
                                        :14
                                    ],
                                    "smiles": "N",
                                    "sample_count": 1,
                                },
                                {
                                    "key": Chem.MolToInchiKey(Chem.MolFromSmiles("O"))[
                                        :14
                                    ],
                                    "smiles": "O",
                                    "sample_count": 5,
                                },
                            ],
                            {},
                        ),
                    ),
                ):
                    result = predict(
                        root / "model.pt",
                        root / "test.parquet",
                        root / "base.csv",
                        root / "out.csv",
                        routing_csv=root / "route.csv",
                        encoder_path=encoder,
                        prefix=2 if adaptive else 5,
                        slots=1,
                        full_rankings=root / "full.json",
                        open_protected=True,
                        frequency_weight=frequency_weight,
                        adaptive_prefix=adaptive,
                        expanded_prefix=expanded,
                        first_gate={"confidence": 0.2, "margin": 0.05}
                        if gate_confidence is not None
                        else None,
                        critic_checkpoint=root / "critic.pt"
                        if gate_confidence is not None
                        else None,
                        critic_weight=0.5 if gate_confidence is not None else 0,
                    )
                ranking = result.smiles.iloc[0].split(";")
                active = allowed or expanded is not None
                protected_prefix = (
                    expanded if expanded and not allowed else 1 if adaptive else 5
                )
                if (
                    gate_confidence is not None
                    and gate_confidence < 0.2
                    and active
                    and not timeout
                ):
                    protected_prefix = 0
                self.assertEqual(ranking[:protected_prefix], base[:protected_prefix])
                self.assertEqual(
                    ranking,
                    base[:protected_prefix]
                    + ["O" if frequency_weight else "N"]
                    + base[protected_prefix:]
                    if active and not timeout
                    else base,
                )
                self.assertEqual(decoder.generate.call_count, int(active))
                audit = pd.read_csv(str(root / "out.csv") + ".generation.csv")
                if timeout:
                    self.assertEqual(audit.status.iloc[0], "budget_retrieval_fallback")

    def test_replay_cli_forwards_open_protected(self):
        from unittest.mock import patch

        from casmi_ml.generation_replay import main

        with (
            patch(
                "sys.argv",
                [
                    "generation_replay",
                    "--generated",
                    "samples.json",
                    "--incumbent",
                    "mass",
                    "--output",
                    "replay",
                    "--open-protected",
                ],
            ),
            patch("casmi_ml.generation_replay.run", return_value={}) as run,
        ):
            main()
        self.assertTrue(run.call_args.args[7])


class DecoderCheckpointPackagingTests(unittest.TestCase):
    def test_selected_decoder_is_packaged_and_wrong_replay_checkpoint_is_rejected(self):
        from unittest.mock import patch

        import torch

        from casmi_ml.data import write_json
        from casmi_ml.experimental_release import package
        from casmi_ml.metfrag import digest

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, round_dir = root / "source", root / "round"
            generator_root = root / "research"
            (generator_root / "generation").mkdir(parents=True)
            source.mkdir()
            encoder = root / "encoder.pt"
            encoder.write_bytes(b"frozen encoder")
            chosen = root / "chosen.pt"
            torch.save(
                {
                    "config": {"encoder_sha256": digest(encoder)},
                    "decoder": {"changed": torch.tensor([1.0])},
                },
                chosen,
            )
            external = root / "external/catalog.parquet"
            external.parent.mkdir()
            external.write_bytes(b"external catalog")
            for name in ["ATTRIBUTION.md", "manifest.json", "zenodo_record.json"]:
                (external.parent / name).write_text("{}")
            write_json(source / "report.json", {})
            write_json(
                source / "protocol.json",
                {"external": {"derived_sha256": digest(external)}},
            )
            samples = (
                generator_root
                / "generation"
                / (
                    "researchdev_samples128_limitall_stable_v2_"
                    + digest(chosen)[:12]
                    + ".json"
                )
            )
            write_json(
                samples, [{"key": str(i), "candidates": []} for i in range(2000)]
            )
            write_json(
                samples.with_suffix(".config.json"),
                {"checkpoint_sha256": digest(chosen)},
            )
            write_json(
                round_dir / "protocol.json",
                {
                    "open_protected": True,
                    "source_directory": str(source),
                    "source_report_sha256": digest(source / "report.json"),
                    "generator_checkpoint": str(chosen),
                    "generator_sha256": digest(chosen),
                    "limit": None,
                    "prefix": 3,
                    "slots": 5,
                },
            )
            replay = {
                "valid": True,
                "molecules": 75,
                "open_protected": True,
                "high_confidence_branch": 25,
                "prefix": 3,
                "slots": 5,
                "generator_sha256": digest(chosen),
                "samples_sha256": digest(samples),
            }
            write_json(round_dir / "replay.json", replay)
            decision = root / "decision.json"
            write_json(
                decision,
                {
                    "direction": "generation_model",
                    "round_directory": str(round_dir),
                    "winner": {"variant": "model", "gate": {"eligible": True}},
                },
            )

            def prepare(output):
                output = Path(output)
                write_json(output / "bundle/deployment_recipe.json", {})
                write_json(output / "dataset/dataset-metadata.json", {})
                write_json(output / "notebook/kernel-metadata.json", {})
                write_json(
                    output / "notebook/casmi_chemistry.ipynb",
                    {
                        "cells": [
                            {"source": "title"},
                            {"source": "casmi_ml.secondary_inference"},
                        ]
                    },
                )

            with (
                patch("casmi_ml.experimental_release.prepare", side_effect=prepare),
                patch("casmi_ml.experimental_release.copy_inference_source"),
                patch("casmi_ml.research_protocol.ROOT", generator_root),
                patch("casmi_ml.research_protocol.ENCODER", encoder),
                patch("casmi_ml.coverage_experiment.EXTERNAL", external),
            ):
                output = root / "release"
                package("decoder", decision, output)
                recipe = json.loads(
                    (output / "bundle/deployment_recipe.json").read_text()
                )
                self.assertEqual(recipe["generation"]["sha256"], digest(chosen))
                self.assertEqual(
                    digest(output / "bundle/generation.pt"), digest(chosen)
                )
                self.assertEqual(recipe["generation"]["prefix"], 3)
                # A frequency release must bind both its measured cache and
                # selected fusion weight to actual unlabeled replay evidence.
                frequency_samples = samples.with_name("frequency_samples.json")
                write_json(frequency_samples, json.loads(samples.read_text()))
                write_json(
                    frequency_samples.with_suffix(".config.json"),
                    {
                        "checkpoint_sha256": digest(chosen),
                        "candidate_statistics": "sample_frequency_and_best_sequence_token_count_v1",
                    },
                )
                protocol = json.loads((round_dir / "protocol.json").read_text())
                frequency_protocol = {
                    **protocol,
                    "generated_path": str(frequency_samples),
                    "variants": {"frequency_1": 1.0},
                }
                write_json(round_dir / "protocol.json", frequency_protocol)
                frequency_replay = {
                    **replay,
                    "frequency_weight": 1.0,
                    "samples_sha256": digest(frequency_samples),
                }
                write_json(round_dir / "replay.json", frequency_replay)
                frequency_decision = root / "frequency_decision.json"
                write_json(
                    frequency_decision,
                    {
                        "direction": "generated_frequency",
                        "round_directory": str(round_dir),
                        "winner": {
                            "variant": "frequency_1",
                            "gate": {"eligible": True},
                        },
                    },
                )
                frequency_output = root / "frequency_release"
                package("frequency", frequency_decision, frequency_output)
                frequency_recipe = json.loads(
                    (frequency_output / "bundle/deployment_recipe.json").read_text()
                )
                self.assertEqual(
                    frequency_recipe["generation"]["frequency_weight"], 1.0
                )
                self.assertEqual(
                    digest(frequency_output / "bundle/generation.pt"), digest(chosen)
                )
                frequency_replay["frequency_weight"] = 0.5
                write_json(round_dir / "replay.json", frequency_replay)
                with self.assertRaises(ValueError):
                    package(
                        "wrongfrequency", frequency_decision, root / "wrongfrequency"
                    )
                self.assertFalse((root / "wrongfrequency").exists())
                critic_path = root / "critic.pt"
                torch.save(
                    {
                        "encoder_sha256": digest(encoder),
                        "architecture": "fingerprint",
                        "state_dict": {},
                    },
                    critic_path,
                )
                write_json(
                    round_dir / "deployment.json",
                    {
                        "protocol_sha256": digest(round_dir / "protocol.json"),
                        "variant": "calibrated_prefix2_slots5",
                        "prefix": 2,
                        "slots": 5,
                        "calibrated": True,
                        "generator_checkpoint": str(chosen),
                        "generator_sha256": digest(chosen),
                        "generated_path": str(frequency_samples),
                        "critic_checkpoint": str(critic_path),
                        "critic_sha256": digest(critic_path),
                        "limit": None,
                    },
                )
                calibrated_replay = {
                    **replay,
                    "prefix": 2,
                    "token_length_exponent": 1.0,
                    "critic_weight": 0.5,
                    "critic_sha256": digest(critic_path),
                    "samples_sha256": digest(frequency_samples),
                }
                write_json(round_dir / "replay.json", calibrated_replay)
                calibrated_decision = root / "calibrated_decision.json"
                write_json(
                    calibrated_decision,
                    {
                        "direction": "generated_position_update",
                        "round_directory": str(round_dir),
                        "winner": {
                            "variant": "calibrated_prefix2_slots5",
                            "gate": {"eligible": True},
                        },
                    },
                )
                calibrated_output = root / "calibrated_release"
                package("calibrated", calibrated_decision, calibrated_output)
                calibrated_recipe = json.loads(
                    (calibrated_output / "bundle/deployment_recipe.json").read_text()
                )
                self.assertEqual(
                    calibrated_recipe["generation"]["token_length_exponent"], 1.0
                )
                self.assertEqual(calibrated_recipe["generation"]["prefix"], 2)
                self.assertEqual(
                    calibrated_recipe["generation"]["critic"]["weight"], 0.5
                )
                self.assertEqual(
                    digest(calibrated_output / "bundle/generated_critic.pt"),
                    digest(critic_path),
                )
                calibrated_replay["critic_weight"] = 0.25
                write_json(round_dir / "replay.json", calibrated_replay)
                with self.assertRaises(ValueError):
                    package(
                        "wrongcalibration",
                        calibrated_decision,
                        root / "wrongcalibration",
                    )
                self.assertFalse((root / "wrongcalibration").exists())
                # Adaptive insertion must be bound to the actual replay rule.
                calibrated_replay["critic_weight"] = 0.5
                calibrated_replay["adaptive_prefix"] = "second_unreferenced"
                write_json(round_dir / "replay.json", calibrated_replay)
                adaptive_deployment = json.loads(
                    (round_dir / "deployment.json").read_text()
                )
                adaptive_deployment.update(
                    variant="second_unreferenced", adaptive_prefix="second_unreferenced"
                )
                write_json(round_dir / "deployment.json", adaptive_deployment)
                adaptive_decision = root / "adaptive_decision.json"
                write_json(
                    adaptive_decision,
                    {
                        "direction": "generated_second_reference",
                        "round_directory": str(round_dir),
                        "winner": {
                            "variant": "second_unreferenced",
                            "gate": {"eligible": True},
                        },
                    },
                )
                adaptive_output = root / "adaptive_release"
                package("adaptive", adaptive_decision, adaptive_output)
                adaptive_recipe = json.loads(
                    (adaptive_output / "bundle/deployment_recipe.json").read_text()
                )
                self.assertEqual(
                    adaptive_recipe["generation"]["adaptive_prefix"],
                    "second_unreferenced",
                )
                self.assertEqual(adaptive_recipe["generation"]["prefix"], 2)
                calibrated_replay["adaptive_prefix"] = None
                write_json(round_dir / "replay.json", calibrated_replay)
                with self.assertRaises(ValueError):
                    package("wrongadaptive", adaptive_decision, root / "wrongadaptive")
                self.assertFalse((root / "wrongadaptive").exists())
                calibrated_replay.update(
                    adaptive_prefix="second_unreferenced",
                    expanded_prefix=2,
                    expanded_branch=25,
                )
                write_json(round_dir / "replay.json", calibrated_replay)
                adaptive_deployment.update(
                    variant="expanded_prefix2", expanded_prefix=2
                )
                write_json(round_dir / "deployment.json", adaptive_deployment)
                expanded_decision = root / "expanded_decision.json"
                write_json(
                    expanded_decision,
                    {
                        "direction": "generated_expanded_route",
                        "round_directory": str(round_dir),
                        "winner": {
                            "variant": "expanded_prefix2",
                            "gate": {"eligible": True},
                        },
                    },
                )
                expanded_output = root / "expanded_release"
                package("expanded", expanded_decision, expanded_output)
                self.assertEqual(
                    json.loads(
                        (expanded_output / "bundle/deployment_recipe.json").read_text()
                    )["generation"]["expanded_prefix"],
                    2,
                )
                calibrated_replay["expanded_prefix"] = None
                write_json(round_dir / "replay.json", calibrated_replay)
                with self.assertRaises(ValueError):
                    package("wrongexpanded", expanded_decision, root / "wrongexpanded")
                self.assertFalse((root / "wrongexpanded").exists())
                calibrated_replay.update(
                    expanded_prefix=2,
                    first_gate={"confidence": 0.2, "margin": 0.05},
                    generated_first_promotions=1,
                )
                write_json(round_dir / "replay.json", calibrated_replay)
                adaptive_deployment.update(
                    variant="confidence0.2_margin0.05",
                    first_gate={"confidence": 0.2, "margin": 0.05},
                )
                write_json(round_dir / "deployment.json", adaptive_deployment)
                gated_decision = root / "gated_decision.json"
                write_json(
                    gated_decision,
                    {
                        "direction": "generated_first_gate",
                        "round_directory": str(round_dir),
                        "winner": {
                            "variant": "confidence0.2_margin0.05",
                            "gate": {"eligible": True},
                        },
                    },
                )
                gated_output = root / "gated_release"
                package("gated", gated_decision, gated_output)
                self.assertEqual(
                    json.loads(
                        (gated_output / "bundle/deployment_recipe.json").read_text()
                    )["generation"]["first_gate"],
                    {"confidence": 0.2, "margin": 0.05},
                )
                calibrated_replay["generated_first_promotions"] = 0
                write_json(round_dir / "replay.json", calibrated_replay)
                with self.assertRaises(ValueError):
                    package("wronggate", gated_decision, root / "wronggate")
                self.assertFalse((root / "wronggate").exists())
                write_json(round_dir / "protocol.json", protocol)
                replay["generator_sha256"] = "wrong checkpoint"
                write_json(round_dir / "replay.json", replay)
                with self.assertRaises(ValueError):
                    package("wrong", decision, root / "invalid_release")
                self.assertFalse((root / "invalid_release").exists())


class PartialPromotionTests(unittest.TestCase):
    def test_front_slots_preserve_retrieval_and_remaining_generated_order(self):
        from casmi_ml.generated_partial_promotion import partial_promotion
        from casmi_ml.generation_slots import insert_generated

        base = ["a", "b", "c", "d"]
        generated = ["a", "x", "x", "y", "z"]
        self.assertEqual(
            partial_promotion(base, generated, 2, 1),
            ["x", "a", "b", "y", "z", "c", "d"],
        )
        self.assertEqual(
            partial_promotion(base, generated, 2, 5),
            insert_generated(base, generated, 0, 5),
        )
        self.assertEqual(partial_promotion(base, ["a", "b"], 2, 1), base)
        for front in [0, 6]:
            with self.assertRaises(ValueError):
                partial_promotion(base, generated, 2, front)


class ConditionContrastTrainingTests(unittest.TestCase):
    def test_training_mass_negatives_are_distinct_and_deterministic(self):
        import numpy as np

        from casmi_ml.generation_condition_contrastive import nearest_mass_negatives

        masses = [100.0, 100.0, 100.0, 102.0]
        keys = ["c", "a", "b", "d"]
        indexes = nearest_mass_negatives(masses, keys)
        self.assertEqual(indexes.tolist(), [2, 2, 1, 0])
        self.assertTrue(np.all(indexes != np.arange(4)))
        for invalid in [[100.0], [100.0, float("nan")]]:
            with self.assertRaises(ValueError):
                nearest_mass_negatives(invalid, ["a", "b"][: len(invalid)])

    def test_sequence_nll_ignores_padding_and_reaches_logits(self):
        import torch

        from casmi_ml.generation_condition_contrastive import sequence_nll

        logits = torch.zeros(2, 3, 4, requires_grad=True)
        target = torch.tensor([[1, 2, 0], [1, 0, 0]])
        value = sequence_nll(logits, target)
        self.assertTrue(
            torch.allclose(value, torch.full((2,), torch.log(torch.tensor(4.0))))
        )
        value.sum().backward()
        self.assertTrue(
            torch.equal(
                logits.grad[target == 0], torch.zeros_like(logits.grad[target == 0])
            )
        )
        self.assertGreater(float(logits.grad.abs().sum()), 0)


class OfflineRuntimeWheelTests(unittest.TestCase):
    def test_python_abi_and_version_select_exactly_one_wheel(self):
        from packaging.tags import Tag

        from casmi_ml.runtime_wheels import compatible_rdkit_wheel

        with tempfile.TemporaryDirectory() as d:
            for abi in ["cp312", "cp313"]:
                (
                    Path(d) / f"rdkit-2026.3.6-{abi}-{abi}-manylinux_2_28_x86_64.whl"
                ).touch()
            for abi in ["cp312", "cp313"]:
                wheel = compatible_rdkit_wheel(
                    d, "2026.3.6", [Tag(abi, abi, "manylinux_2_28_x86_64")]
                )
                self.assertIn(abi, wheel.name)
            for version, abi in [("2026.3.5", "cp313"), ("2026.3.6", "cp314")]:
                with self.assertRaises(ValueError):
                    compatible_rdkit_wheel(
                        d, version, [Tag(abi, abi, "manylinux_2_28_x86_64")]
                    )


class GeneratedLikelihoodRatioTests(unittest.TestCase):
    def test_score_masks_forbidden_tokens_and_padding(self):
        import math

        import torch

        from casmi_ml.generated_likelihood_ratio import log_likelihood

        logits = torch.zeros(2, 3, 5)
        logits[:, :, [0, 1, 3]] = 100
        target = torch.tensor([[4, 2, 0], [4, 2, 4]])
        score = log_likelihood(logits, target)
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.allclose(score, torch.full((2,), -math.log(2))))
        # Forbidden-token logits cannot change legal sequence likelihood.
        logits[:, :, [0, 1, 3]] = -100
        self.assertTrue(torch.equal(score, log_likelihood(logits, target)))


class ChEMBLCatalogImportTests(unittest.TestCase):
    def test_structure_normalization_preserves_identity_and_rejects_invalid_sources(
        self,
    ):
        from rdkit import Chem

        from casmi_ml.chembl_catalog import normalize

        molecule = Chem.MolFromSmiles("C[C@H](O)C(=O)O")
        row = {
            "chembl_id": "example",
            "canonical_smiles": "C[C@H](O)C(=O)O",
            "standard_inchi_key": Chem.MolToInchiKey(molecule),
        }
        record, reason = normalize(row)
        self.assertEqual(reason, "retained")
        self.assertNotIn("@", record["normalized_smiles"])
        self.assertEqual(record["inchikey14"], row["standard_inchi_key"][:14])
        for smiles in ["CC.O", "[NH4+]", "[13CH4]", "*CC"]:
            row["canonical_smiles"] = smiles
            self.assertIsNone(normalize(row)[0])
        row["canonical_smiles"] = "CCO"
        self.assertEqual(normalize(row)[1], "identity_disagreement")


class CatalogCriticCacheTests(unittest.TestCase):
    def test_cache_binds_complete_scored_candidate_list(self):
        from casmi_ml.chembl_critic_slots import score_cache_key

        value = score_cache_key("query", "first", ["a", "b"])
        self.assertEqual(value, score_cache_key("query", "first", ["a", "b"]))
        for query, first, candidates in [
            ("other", "first", ["a", "b"]),
            ("query", "other", ["a", "b"]),
            ("query", "first", ["a", "c"]),
            ("query", "first", ["b", "a"]),
        ]:
            self.assertNotEqual(value, score_cache_key(query, first, candidates))
