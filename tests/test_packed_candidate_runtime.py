import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_runtime import resolve


class PackedRuntimeSelectionTests(unittest.TestCase):
    def test_no_activation_or_remote_context_never_selects_cache(self):
        c = SimpleNamespace(local_only=lambda: True, read=dict)
        self.assertIsNone(resolve(c, "catalog"))
        c.local_only = lambda: False
        c.read = lambda: {"local_candidate_runtime": {"enabled": True}}
        self.assertIsNone(resolve(c, "catalog"))

    def test_active_cache_requires_current_catalog_source_and_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "verified": True,
                        "development_budget_fallbacks": 0,
                        "incremental_new_failure_contexts": 0,
                    }
                )
            )
            value = {
                "enabled": True,
                "catalog_sha256": "catalog",
                "source_sha256": digest("casmi_ml/packed_candidate_cache.py"),
                "control_directory": str(root),
                "control_report_sha256": digest(report),
                "store": "cache",
            }
            c = SimpleNamespace(
                local_only=lambda: True, read=lambda: {"local_candidate_runtime": value}
            )
            self.assertTrue(callable(resolve(c, "catalog")))
            with self.assertRaisesRegex(ValueError, "catalog differs"):
                resolve(c, "wrong")
            report.write_text("{}")
            with self.assertRaisesRegex(ValueError, "control changed"):
                resolve(c, "catalog")
