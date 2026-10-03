import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml.candidate_compatibility_activation import activate, resolve_score
from casmi_ml.metfrag import digest


class CandidateActivationTests(unittest.TestCase):
    def fixture(self, root):
        out = root / "rounds/winner"
        zero = root / "rounds/0299_candidate_compatibility_zero_control"
        for directory in (out, zero):
            directory.mkdir(parents=True)
        model = root / "model.json"
        model.write_text("{}")
        protocol = {
            **{
                field: digest(source)
                for field, source in {
                    "source_sha256": "casmi_ml/candidate_compatibility_system.py",
                    "inference_source_sha256": "casmi_ml/chembl_routed_inference.py",
                    "compatibility_source_sha256": "casmi_ml/candidate_diagnostic_compatibility.py",
                    "evidence_source_sha256": "casmi_ml/diagnostic_residual.py",
                }.items()
            },
            "control": False,
            "config": {"encoder": "encoder", "critic": "critic"},
            "model": str(model),
            "model_sha256": digest(model),
        }
        (out / "protocol.json").write_text(json.dumps(protocol))
        (zero / "protocol.json").write_text(json.dumps(dict(protocol, control=True)))
        (zero / "report.json").write_text(json.dumps({"control_passed": True}))
        report = {
            "development_budget_fallbacks": 0,
            "incremental_new_failure_contexts": 0,
        }
        (out / "report.json").write_text(json.dumps(report))
        (out / "external_config.json").write_text(json.dumps(protocol["config"]))
        (out / "selected_rankings.json").write_text("{}")
        state = {
            "development_best": {
                "round": "winner",
                "directory": str(out),
                "variant": "candidate_compatibility",
            }
        }
        c = SimpleNamespace(
            root=root,
            local_only=lambda: True,
            read=lambda: state,
            change=lambda fn: fn(state),
        )
        decision = {"winner": {"variant": "candidate_compatibility", "metrics": {}}}
        return c, out, decision, state

    def test_complete_winner_activates_and_resolves_bound_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, out, decision, state = self.fixture(Path(tmp))
            fn = lambda *args: None
            with (
                patch(
                    "casmi_ml.candidate_compatibility_activation.load_model",
                    return_value=(fn, {"mask_chemical_channels": False}),
                ),
                patch("casmi_ml.candidate_compatibility_activation.write_json"),
            ):
                proof = activate(c, out, decision)
                self.assertEqual(
                    state["local_verified_inference"]["scientific_round"], "winner"
                )
                self.assertIs(
                    resolve_score({"encoder": "encoder", "critic": "critic"}, proof), fn
                )
            Path(proof["candidate_compatibility"]["model"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "weights changed"):
                resolve_score({"encoder": "encoder", "critic": "critic"}, proof)

    def test_missing_control_and_incomplete_physical_evidence_cannot_activate(self):
        for field in (
            "development_budget_fallbacks",
            "incremental_new_failure_contexts",
            "control_passed",
        ):
            with tempfile.TemporaryDirectory() as tmp:
                c, out, decision, state = self.fixture(Path(tmp))
                target = (
                    c.root
                    / "rounds/0299_candidate_compatibility_zero_control/report.json"
                    if field == "control_passed"
                    else out / "report.json"
                )
                report = json.loads(target.read_text())
                report[field] = False if field == "control_passed" else 1
                target.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    activate(c, out, decision)
                self.assertNotIn("local_verified_inference", state)

    def test_changed_source_and_wrong_current_best_cannot_activate(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, out, decision, state = self.fixture(Path(tmp))
            protocol = json.loads((out / "protocol.json").read_text())
            protocol["compatibility_source_sha256"] = "changed"
            (out / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "source changed"):
                activate(c, out, decision)
            self.assertNotIn("local_verified_inference", state)
        with tempfile.TemporaryDirectory() as tmp:
            c, out, decision, state = self.fixture(Path(tmp))
            state["development_best"]["directory"] = "other"
            with self.assertRaisesRegex(ValueError, "current complete winner"):
                activate(c, out, decision)

    def test_existing_original_runtime_is_unchanged_without_candidate_spec(self):
        self.assertIsNone(resolve_score({}, {}))
        self.assertIsNone(activate(None, None, {"winner": None}))
        with self.assertRaisesRegex(ValueError, "original frozen critic"):
            resolve_score(
                {"query_residual": True},
                {"candidate_compatibility": {"model": "unused"}},
            )
