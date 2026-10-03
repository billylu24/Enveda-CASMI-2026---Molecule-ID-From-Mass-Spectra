import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.research_loop import Controller


class HighFragmentControllerTests(unittest.TestCase):
    def controller(self, root):
        config = json.loads(Path("configs/research_loop.json").read_text())
        config.update(root=str(root / "state"), automatic_github_sync=False)
        write_json(root / "config.json", config)
        controller = Controller(root / "config.json")
        report = root / "experiment/report.json"
        write_json(
            report.parent / "decision.json",
            {
                "winner": {
                    "variant": "high_relative_fragment",
                    "gate": {"eligible": True},
                },
            },
        )
        controller.register_round("high", "chembl_high_fragment", [], report)
        controller.mark_round(
            "high", status="eligible", decision=str(report.parent / "decision.json")
        )
        return controller, report.parent

    def test_eligible_high_extension_uses_release_path(self):
        with tempfile.TemporaryDirectory() as directory:
            c, _ = self.controller(Path(directory))
            with patch.object(
                c, "release_round", return_value="notebook_running"
            ) as release:
                self.assertEqual(c.complete_round("high"), "publication_waiting")
            release.assert_called_once_with("high")
            self.assertEqual(c.read()["rounds"][0]["status"], "eligible")

    def test_replay_failure_blocks_packaging_and_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            c, _ = self.controller(Path(directory))
            with (
                patch.object(c, "job", return_value="budget_exhausted") as job,
                patch("casmi_ml.chembl_routed_release.prepare") as prepare,
                patch.object(c, "publish_prepared") as publish,
            ):
                self.assertEqual(c.release_round("high"), "budget_exhausted")
            self.assertIn("casmi_ml.chembl_high_fragment_replay", job.call_args.args[0])
            prepare.assert_not_called()
            publish.assert_not_called()
            self.assertTrue(c.read()["rounds"][0]["requires_platform_verification"])

    def test_high_release_uses_matching_builder_gpu_verify_and_platform_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            c, experiment = self.controller(Path(directory))
            write_json(experiment / "replay.json", {"valid": True, "molecules": 75})
            release = c.root / "releases/high"

            def prepare(source, output):
                self.assertEqual(source, experiment)
                output.mkdir(parents=True)

            with (
                patch(
                    "casmi_ml.chembl_routed_release.prepare", side_effect=prepare
                ) as builder,
                patch("casmi_ml.experimental_release.package") as old_builder,
                patch.object(c, "job", return_value="complete") as job,
                patch.object(
                    c, "publish_prepared", return_value={"status": "notebook_running"}
                ) as publish,
            ):
                self.assertEqual(c.release_round("high"), "notebook_running")
            builder.assert_called_once_with(experiment, release)
            old_builder.assert_not_called()
            self.assertEqual(job.call_args.args[0][0], ".venv-gpu/bin/python")
            self.assertEqual(job.call_args.args[2], 1800)
            self.assertTrue(c.read()["rounds"][0]["requires_platform_verification"])
            publish.assert_called_once_with("high", release)
