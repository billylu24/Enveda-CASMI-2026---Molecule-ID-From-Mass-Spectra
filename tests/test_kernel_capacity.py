import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity


class KernelCapacityTests(unittest.TestCase):
    def test_explicit_gpu_capacity_rejection_preserves_prepared_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path("configs/research_loop.json").read_text())
            config["root"] = str(root / "state")
            write_json(root / "config.json", config)
            controller = Controller(root / "config.json")
            release = root / "release"
            (release / "bundle").mkdir(parents=True)
            code = release / "bundle/code.py"
            code.write_text("pass")
            sums = {"code.py": digest(code)}
            write_json(release / "bundle/SHA256SUMS.json", sums)
            write_json(release / "notebook/kernel-metadata.json", {"id": "owner/new"})
            write_json(release / "notebook/code.ipynb", {"cells": []})
            write_json(release / "dataset/dataset-metadata.json", {"id": "owner/data"})
            identity = release_identity(release, sums, "selected")
            write_json(
                release / "verification.json",
                {
                    "identity": identity,
                    "valid": True,
                    "molecules": 400,
                    "seconds": 1,
                    "peak_rss_mib": 1,
                },
            )
            write_json(
                root / "decision.json",
                {"winner": {"variant": "selected", "gate": {"eligible": True}}},
            )
            controller.register_round("round", "mass", [], root / "report.json")
            controller.mark_round(
                "round",
                status="eligible",
                decision=str(root / "decision.json"),
                dataset_uploaded=True,
            )
            api = MagicMock()
            api.dataset_status.return_value = "ready"
            response = SimpleNamespace(
                error="Maximum batch GPU session count of 2 reached.",
                ref="",
                version_number=None,
            )
            api.kernels_push.return_value = response
            with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=api):
                self.assertEqual(
                    controller.publish_prepared("round", release)["status"],
                    "notebook_running",
                )
                row = controller.read()["rounds"][0]
                self.assertEqual(row["status"], "eligible")
                self.assertIsNone(row.get("remote_release"))
                self.assertIsNone(controller.read()["pending_submission"])
                api.competition_submit_code.assert_not_called()
                response.ref = "owner/ambiguous"
                with self.assertRaises(ValueError):
                    controller.publish_prepared("round", release)
