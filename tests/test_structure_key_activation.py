import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml.structure_key_activation import activate


class StructureKeyActivationTests(unittest.TestCase):
    def test_partial_or_changed_incumbent_never_mutates_source_or_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = {"inference_sha256": "old"}
            (root / "protocol.json").write_text(json.dumps({"incumbent_proof": proof}))
            (root / "report.json").write_text(json.dumps({"verified": False}))
            c = SimpleNamespace(
                local_only=lambda: True,
                read=lambda: {"local_verified_inference": proof},
                change=lambda f: self.fail("State mutation"),
            )
            with (
                patch.object(
                    Path, "write_bytes", side_effect=AssertionError("Source mutation")
                ),
                self.assertRaisesRegex(ValueError, "unchanged-incumbent"),
            ):
                activate(c, root)
            (root / "report.json").write_text(
                json.dumps(
                    {
                        "verified": True,
                        "development_budget_fallbacks": 0,
                        "incremental_new_failure_contexts": 0,
                    }
                )
            )
            c.read = lambda: {"local_verified_inference": {"inference_sha256": "new"}}
            with (
                patch.object(
                    Path, "write_bytes", side_effect=AssertionError("Source mutation")
                ),
                self.assertRaisesRegex(ValueError, "unchanged-incumbent"),
            ):
                activate(c, root)
