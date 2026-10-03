import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml.diagnostic_query_activation import activate
from casmi_ml.metfrag import digest


class QueryActivationTests(unittest.TestCase):
    def fixture(self, root):
        output = root / "rounds/winner"
        zero = root / "rounds/0244_diagnostic_query_zero_control"
        output.mkdir(parents=True)
        zero.mkdir(parents=True)
        protocol = {
            field: digest(name)
            for name, field in [
                ("casmi_ml/chembl_routed_inference.py", "inference_source_sha256"),
                ("casmi_ml/diagnostic_query_ranker.py", "query_source_sha256"),
                ("casmi_ml/diagnostic_query_residual.py", "query_residual_sha256"),
            ]
        }
        report = {
            "development_budget_fallbacks": 0,
            "incremental_new_failure_contexts": 0,
        }
        for directory in [output, zero]:
            (directory / "protocol.json").write_text(json.dumps(protocol))
        (output / "report.json").write_text(json.dumps(report))
        (output / "selected_rankings.json").write_text("{}")
        (output / "external_config.json").write_text("{}")
        (zero / "report.json").write_text(json.dumps({"control_passed": True}))
        state = {
            "development_best": {
                "directory": str(output),
                "variant": "chemical_query_residual",
                "round": "winner",
            }
        }
        c = SimpleNamespace(
            root=root,
            local_only=lambda: True,
            read=lambda: state,
            change=lambda fn: fn(state),
        )
        result = {
            "winner": {
                "variant": "chemical_query_residual",
                "metrics": {"unknown": {"mrr25": 0.05}},
            }
        }
        return c, output, result, state

    def test_activation_requires_complete_matching_control_and_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            c, output, result, state = self.fixture(Path(directory))
            with patch("casmi_ml.diagnostic_query_activation.write_json"):
                proof = activate(c, output, result)
            self.assertEqual(
                state["local_verified_inference"]["evidence_round"], "winner"
            )
            self.assertFalse(proof["remote_publication"])
            report = json.loads((output / "report.json").read_text())
            report["incremental_new_failure_contexts"] = 1
            (output / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Incomplete query evidence"):
                activate(c, output, result)

    def test_runtime_source_bindings_survive_a_new_scientific_query_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            c, output, result, state = self.fixture(Path(directory))
            protocol = json.loads((output / "protocol.json").read_text())
            protocol["config"] = {
                "candidate_order_runtime": "array_v1",
                "structure_identity_runtime": "cache_v1",
                "structure_identity_source_sha256": digest(
                    "casmi_ml/cached_structure_keys.py"
                ),
            }
            (output / "protocol.json").write_text(json.dumps(protocol))
            with patch("casmi_ml.diagnostic_query_activation.write_json"):
                proof = activate(c, output, result)
            self.assertEqual(
                state["local_verified_inference"]["identity_source_sha256"],
                proof["identity_source_sha256"],
            )
            self.assertEqual(
                proof["identity_source_sha256"],
                digest("casmi_ml/cached_structure_keys.py"),
            )
            self.assertEqual(
                proof["array_source_sha256"],
                digest("casmi_ml/candidate_array_order.py"),
            )

    def test_transform_source_requires_matching_zero_control(self):
        with tempfile.TemporaryDirectory() as directory:
            c, output, result, state = self.fixture(Path(directory))
            protocol = json.loads((output / "protocol.json").read_text())
            protocol["strength_transform_sha256"] = digest(
                "casmi_ml/diagnostic_strength_transform.py"
            )
            (output / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "zero control source differs"):
                activate(c, output, result)
            self.assertNotIn("local_verified_inference", state)
            protocol["strength_transform_sha256"] = "changed"
            (output / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "transform source changed"):
                activate(c, output, result)

    def test_missing_control_never_activates_a_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            c, output, result, state = self.fixture(Path(directory))
            (
                c.root / "rounds/0244_diagnostic_query_zero_control/report.json"
            ).write_text(json.dumps({"control_passed": False}))
            with self.assertRaisesRegex(ValueError, "zero-residual"):
                activate(c, output, result)
            self.assertNotIn("local_verified_inference", state)
