import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from casmi_ml.array_order_full_replay import activate


class ArrayActivationTests(unittest.TestCase):
    def test_incomplete_or_different_incumbent_never_mutates_source_or_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"verified": False}
            proof = {"inference_sha256": "old"}
            (root / "report.json").write_text(json.dumps(report))
            (root / "protocol.json").write_text(json.dumps({"incumbent_proof": proof}))
            c = SimpleNamespace(
                local_only=lambda: True,
                read=lambda: {"local_verified_inference": proof},
                change=lambda fn: self.fail("State mutation"),
            )
            with patch.object(
                Path, "write_bytes", side_effect=AssertionError("Source mutation")
            ), self.assertRaisesRegex(ValueError, "unchanged-incumbent"):
                activate(c, root)
            report["verified"] = True
            (root / "report.json").write_text(json.dumps(report))
            c.read = lambda: {
                "local_verified_inference": {"inference_sha256": "new-incumbent"}
            }
            with patch.object(
                Path, "write_bytes", side_effect=AssertionError("Source mutation")
            ), self.assertRaisesRegex(ValueError, "unchanged-incumbent"):
                activate(c, root)
