import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml.diagnostic_system_evaluation import resolve_local_incremental
from casmi_ml.metfrag import digest


class LocalIncrementalSelectionTests(unittest.TestCase):
    def test_controlled_failure_context_survives_runtime_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            control, audit = root / "control", root / "audit"
            control.mkdir()
            audit.mkdir()
            (audit / "report.json").write_text(json.dumps({"control_passed": True}))
            (control / "report.json").write_text(
                json.dumps({"development_budget_fallbacks": 0})
            )
            (control / "protocol.json").write_text(
                json.dumps(
                    {
                        "control": True,
                        "control_reference": str(audit),
                        "development_incremental_source_sha256": digest(
                            "casmi_ml/development_incremental_metfrag.py"
                        ),
                    }
                )
            )
            failure = control / "private_incremental_failures.json"
            failure.write_text(json.dumps(["exact-request.threads2"]))
            runtime = {
                "enabled": True,
                "control_directory": str(control),
                "benchmark_directory": "benchmark",
                "failures_sha256": digest(failure),
            }
            fake = SimpleNamespace(
                local_only=lambda: True,
                read=lambda: {"local_development_incremental": runtime},
            )
            with patch("casmi_ml.research_loop.Controller", return_value=fake):
                benchmark, hashes, selected = resolve_local_incremental("paired", None)
                self.assertEqual(benchmark, "benchmark")
                self.assertEqual(hashes, {"exact-request.threads2"})
                self.assertEqual(selected["failures_sha256"], digest(failure))
                failure.write_text(json.dumps(["new-request.threads2"]))
                with self.assertRaisesRegex(ValueError, "failure context changed"):
                    resolve_local_incremental("paired", None)

    def test_explicit_benchmark_has_no_unverified_inherited_failures(self):
        benchmark, hashes, selected = resolve_local_incremental("paired", "explicit")
        self.assertEqual(benchmark, "explicit")
        self.assertEqual(hashes, set())
        self.assertIsNone(selected)
