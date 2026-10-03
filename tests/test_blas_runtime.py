import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml.blas_runtime import activate, resolve
from casmi_ml.metfrag import digest


class BlasRuntimeTests(unittest.TestCase):
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
        protocol = {
            "source_sha256": digest("casmi_ml/blas_runtime_control.py"),
            "entry_source_sha256": digest("casmi_ml/local_verified_extension.py"),
            "baseline_proof": {"scientific_round": "best"},
        }
        bench_report = {
            "windows": 200,
            "candidate_ranking_mismatches": 0,
            "score_seconds_two_repetitions": {"blas1": 1, "blas24": 10},
        }
        for path, value in [
            (control / "report.json", report),
            (bench / "report.json", bench_report),
            (bench / "protocol.json", {"blas_info": []}),
        ]:
            path.write_text(json.dumps(value))
        protocol["benchmark_report_sha256"] = digest(bench / "report.json")
        (control / "protocol.json").write_text(json.dumps(protocol))
        state = {"development_best": {"round": "best"}}
        c = SimpleNamespace(
            local_only=lambda: True, read=lambda: state, change=lambda fn: fn(state)
        )
        return c, control, bench, state

    def test_source_bound_full_control_activates_and_restores_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, control, bench, state = self.fixture(Path(tmp))
            with patch("casmi_ml.blas_runtime.write_json"):
                activate(c, control, bench)
            manager = SimpleNamespace(__enter__=lambda: None)
            with (
                patch("casmi_ml.blas_runtime.threadpool_info", return_value=[]),
                patch(
                    "casmi_ml.blas_runtime.threadpool_limits", return_value=manager
                ) as limit,
            ):
                self.assertIs(resolve(c), manager)
                limit.assert_called_once_with(limits=1, user_api="blas")
            (control / "report.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                resolve(c)
            self.assertTrue(state["local_blas_runtime"]["enabled"])

    def test_partial_control_or_no_speedup_cannot_activate(self):
        for which in ["partial", "failure", "slow"]:
            with tempfile.TemporaryDirectory() as tmp:
                c, control, bench, state = self.fixture(Path(tmp))
                path = (
                    bench / "report.json"
                    if which == "slow"
                    else control / "report.json"
                )
                report = json.loads(path.read_text())
                if which == "partial":
                    report["known"]["full_rank_matches"] = 1339
                elif which == "failure":
                    report["incremental_new_failure_contexts"] = 1
                else:
                    report["score_seconds_two_repetitions"]["blas1"] = 11
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    activate(c, control, bench)
                self.assertNotIn("local_blas_runtime", state)

    def test_library_change_rejected_and_new_science_requires_new_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, control, bench, state = self.fixture(Path(tmp))
            with patch("casmi_ml.blas_runtime.write_json"):
                activate(c, control, bench)
            library = {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "changed",
                "version": "1",
                "architecture": "x",
                "filepath": "path",
            }
            with patch("casmi_ml.blas_runtime.threadpool_info", return_value=[library]):
                with self.assertRaisesRegex(ValueError, "library identity"):
                    resolve(c)
            state["development_best"]["round"] = "new_best"
            with resolve(c):
                pass
            with patch("casmi_ml.blas_runtime.threadpool_limits") as limited:
                with resolve(c):
                    pass
                limited.assert_not_called()

    def test_disabled_mode_and_nonlocal_context_leave_threads_unchanged(self):
        for c in [
            SimpleNamespace(local_only=lambda: True, read=dict),
            SimpleNamespace(
                local_only=lambda: False,
                read=lambda: {"local_blas_runtime": {"enabled": True}},
            ),
        ]:
            with patch("casmi_ml.blas_runtime.threadpool_limits") as limited:
                with resolve(c):
                    pass
                limited.assert_not_called()
