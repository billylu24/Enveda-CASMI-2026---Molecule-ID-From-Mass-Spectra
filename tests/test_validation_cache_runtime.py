import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml.cached_submission_validation import validate_submission
from casmi_ml.metfrag import digest
from casmi_ml.validation_cache_runtime import activate, resolve


class ValidationCacheRuntimeTests(unittest.TestCase):
    def fixture(self, root):
        control, bench = root / "control", root / "benchmark"
        control.mkdir()
        bench.mkdir()
        report = {
            "verified": True,
            "development_budget_fallbacks": 0,
            "incremental_new_failure_contexts": 0,
            "unknown": {"full_rank_matches": 2000},
            "known": {"full_rank_matches": 1340},
        }
        proof = {"scientific_round": "best", "evidence_round": "existing"}
        protocol = {
            "baseline_proof": proof,
            **{
                field: digest(path)
                for field, path in {
                    "source_sha256": "casmi_ml/validation_cache_full_control.py",
                    "entry_source_sha256": "casmi_ml/local_verified_extension.py",
                    "cache_source_sha256": "casmi_ml/cached_submission_validation.py",
                    "original_validator_source_sha256": "casmi_ml/inference.py",
                }.items()
            },
        }
        bench_report = {
            "verified_all_actual_input_and_output_rows": True,
            "timings": [
                {"implementation": "legacy", "seconds": 10},
                {"implementation": "cache", "seconds": 1},
            ],
        }
        (control / "report.json").write_text(json.dumps(report))
        (bench / "report.json").write_text(json.dumps(bench_report))
        protocol["benchmark_report_sha256"] = digest(bench / "report.json")
        (control / "protocol.json").write_text(json.dumps(protocol))
        state = {
            "development_best": {"round": "best"},
            "local_verified_inference": proof,
        }
        c = SimpleNamespace(
            local_only=lambda: True, read=lambda: state, change=lambda fn: fn(state)
        )
        return c, control, bench, state

    def test_complete_control_activates_and_returns_the_verified_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, control, bench, state = self.fixture(Path(tmp))
            with patch("casmi_ml.validation_cache_runtime.write_json"):
                activate(c, control, bench)
            self.assertIs(resolve(c), validate_submission)
            (control / "report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "control changed"):
                resolve(c)
            self.assertTrue(state["local_validation_runtime"]["enabled"])

    def test_partial_failed_or_stale_science_never_activates(self):
        for which in ["partial", "failure", "stale", "slow"]:
            with tempfile.TemporaryDirectory() as tmp:
                c, control, bench, state = self.fixture(Path(tmp))
                if which == "stale":
                    state["local_verified_inference"] = {"scientific_round": "other"}
                elif which == "slow":
                    v = json.loads((bench / "report.json").read_text())
                    v["timings"][1]["seconds"] = 11
                    (bench / "report.json").write_text(json.dumps(v))
                else:
                    v = json.loads((control / "report.json").read_text())
                    if which == "partial":
                        v["known"]["full_rank_matches"] = 1339
                    else:
                        v["incremental_new_failure_contexts"] = 1
                    (control / "report.json").write_text(json.dumps(v))
                with self.assertRaises(ValueError):
                    activate(c, control, bench)
                self.assertNotIn("local_validation_runtime", state)

    def test_new_scientific_best_falls_back_and_source_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, control, bench, state = self.fixture(Path(tmp))
            with patch("casmi_ml.validation_cache_runtime.write_json"):
                activate(c, control, bench)
            state["development_best"]["round"] = "other"
            self.assertIsNone(resolve(c))
            state["development_best"]["round"] = "best"
            state["local_validation_runtime"]["cache_source_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "source changed"):
                resolve(c)

    def test_without_enabled_local_proof_returns_original_path(self):
        c = SimpleNamespace(local_only=lambda: True, read=dict)
        self.assertIsNone(resolve(c))
