import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from casmi_ml.completed_system_reports import completed_system_report


class CompleteSystemReportTests(unittest.TestCase):
    def test_complete_retry_is_selected_over_incomplete_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds = []
            for i, budget in enumerate([19, 0]):
                name = "round" if i == 0 else "round_cache_completion1"
                path = root / name
                path.mkdir()
                report = path / "report.json"
                report.write_text(
                    json.dumps(
                        {
                            "development_budget_fallbacks": budget,
                            "incremental_new_failure_contexts": 0,
                        }
                    )
                )
                rounds.append(
                    {"id": name, "report": str(report), "local_recorded": True}
                )
            c = SimpleNamespace(read=lambda: {"rounds": rounds})
            self.assertEqual(
                completed_system_report(c, "round"), root / "round_cache_completion1"
            )
            rounds[1]["local_recorded"] = False
            self.assertIsNone(completed_system_report(c, "round"))

    def test_new_failure_is_incomplete_even_without_budget_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "development_budget_fallbacks": 0,
                        "incremental_new_failure_contexts": 1,
                    }
                )
            )
            c = SimpleNamespace(
                read=lambda: {
                    "rounds": [
                        {"id": "round", "report": str(report), "local_recorded": True}
                    ]
                }
            )
            self.assertIsNone(completed_system_report(c, "round"))

    def test_failed_round_is_not_silently_waited_forever(self):
        c = SimpleNamespace(
            read=lambda: {"rounds": [{"id": "round", "status": "failed"}]}
        )
        with self.assertRaisesRegex(RuntimeError, "failed"):
            completed_system_report(c, "round")
