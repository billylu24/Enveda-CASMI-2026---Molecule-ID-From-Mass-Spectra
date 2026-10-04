import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml import overlap_request_runtime_audit as audit


class OverlapRequestAuditTest(unittest.TestCase):
    def exercise(self, failure=False):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output" / "submission.csv"
            summary = Path(folder) / "summary.json"
            request = (
                Path(folder) / "abcdef.json",
                {"a": "CCO"},
                [(42.0, 0.5)],
                46.0,
                "[M+H]+",
            )
            pool = SimpleNamespace(
                parameters={"timeout": 60, "threads": 2},
                deadline=time.monotonic() + 1200,
                local=SimpleNamespace(),
            )
            result = {"status": "complete", "scores": {"a": 123.0}}

            def perform(obj, req):
                self.assertIs(obj, pool)
                self.assertEqual(req, request)
                self.assertEqual(obj.parameters, {"timeout": 60, "threads": 2})
                if failure:
                    raise ValueError("original timeout")
                return result

            def run(*args):
                audit.physical.FragmentPrefetch.record(pool, None, *request)
                value = audit.physical.FragmentPrefetch.perform(pool, request)
                Path(str(output) + ".report.json").write_text("{}")
                self.assertIs(value, result)
                return value

            with (
                patch.object(
                    audit.physical.FragmentPrefetch, "record", return_value=result
                ),
                patch.object(
                    audit.physical.FragmentPrefetch, "perform", side_effect=perform
                ),
                patch.object(audit, "original_run", side_effect=run),
            ):
                if failure:
                    with self.assertRaisesRegex(ValueError, "original timeout"):
                        audit.measure("recipe", "data", "coconut", output, summary)
                else:
                    self.assertIs(
                        audit.measure("recipe", "data", "coconut", output, summary),
                        result,
                    )
            observed = json.loads(summary.read_text())
            self.assertEqual(observed["success"], not failure)
            self.assertFalse(observed["competition_submission"])
            self.assertEqual(
                observed["physical_requests"][0]["configured_request_timeout"], 60
            )
            self.assertEqual(
                observed["physical_requests"][0]["status"],
                "exception" if failure else "complete",
            )
            self.assertTrue(summary.with_name("request_events.json").exists())

    def test_return_values_and_request_parameters_unchanged(self):
        self.exercise()

    def test_original_failure_persisted_and_propagated(self):
        self.exercise(True)
