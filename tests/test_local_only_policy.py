import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.research_loop import Controller


class LocalOnlyPolicyTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(Path("configs/research_loop.json").read_text())
        config["root"] = str(root / "state")
        write_json(root / "config.json", config)
        c = Controller(root / "config.json")
        c.change(lambda s: s.update(local_only=True))
        c.register_round("local", "mass", [], root / "report.json")
        c.mark_round("local", status="eligible")
        return c

    def test_restart_never_uploads_submits_or_pushes(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.fixture(Path(directory))
            c = Controller(c.config_path)
            with (
                patch("kaggle.api.kaggle_api_extended.KaggleApi") as api,
                patch("casmi_ml.research_loop.subprocess.check_output") as git,
            ):
                self.assertEqual(c.release_round("local"), "local_only")
                self.assertEqual(
                    c.publish_prepared("local", Path(directory) / "nonexistent"),
                    {"status": "local_only"},
                )
                self.assertEqual(c.refresh_kaggle(), [])
                self.assertEqual(c.import_bootstrap(), [])
                self.assertEqual(c.sync_github("local", []), "local_only")
                with self.assertRaisesRegex(RuntimeError, "Local-only"):
                    c.reserve_submission("identity", 1, "local")
                api.assert_not_called()
                git.assert_not_called()

    def test_completed_local_round_does_not_spin_on_eligible_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            c = self.fixture(Path(directory))
            with (
                patch.object(c, "release_round") as release,
                patch.object(c, "sync_github") as sync,
            ):
                self.assertEqual(c.step(), "round_complete")
                self.assertEqual(c.step(), "research_needed")
                release.assert_not_called()
                sync.assert_not_called()
            self.assertTrue(c.read()["rounds"][0]["local_recorded"])
            self.assertEqual(c.read()["rounds"][0]["status"], "eligible")
