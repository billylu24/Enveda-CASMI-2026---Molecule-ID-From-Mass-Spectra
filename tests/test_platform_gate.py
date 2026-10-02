import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity


class PlatformGateTests(unittest.TestCase):
    def test_runtime_migration_cannot_submit_before_bound_full_platform_check(self):
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
                decision=str(root / "decision.json"),
                remote_release={"kernel": "owner/new", "version": 1},
                requires_platform_verification=True,
            )
            api = MagicMock()
            api.kernels_status.return_value = {"status": "COMPLETE"}
            api.competition_submit_code.return_value.ref = 123
            platform = {
                "identity": identity,
                "valid": True,
                "molecules": 400,
                "local_full_top25_matches": 399,
                "seconds": 1,
                "parent_peak_rss_mib": 1,
            }
            with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=api):
                self.assertEqual(
                    controller.publish_prepared("round", release)["status"],
                    "notebook_running",
                )
                write_json(release / "kaggle_verification.json", platform)
                with self.assertRaises(ValueError):
                    controller.publish_prepared("round", release)
                api.competition_submit_code.assert_not_called()
                platform["local_full_top25_matches"] = 400
                write_json(release / "kaggle_verification.json", platform)
                self.assertEqual(
                    controller.publish_prepared("round", release)["id"], 123
                )
            api.competition_submit_code.assert_called_once()
