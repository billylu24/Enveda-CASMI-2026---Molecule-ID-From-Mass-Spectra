import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.research_loop import Controller
from casmi_ml.runtime_migration import activate_round


class RuntimeMigrationTests(unittest.TestCase):
    def setup_case(self, root):
        config = json.loads(Path("configs/research_loop.json").read_text())
        config["root"] = str(root / "state")
        write_json(root / "config.json", config)
        controller = Controller(root / "config.json")
        old, new = root / "old", root / "new"
        for release in (old, new):
            (release / "local_output").mkdir(parents=True)
            pd.DataFrame({"id": range(400), "rank": ["a"] * 400}).to_csv(
                release / "local_output/submission.csv", index=False
            )
            write_json(
                release / "bundle/SHA256SUMS.json",
                {"encoder.pt": "frozen", "generation.pt": "frozen2"},
            )
            write_json(release / "bundle/deployment_recipe.json", {"frozen": True})
        write_json(old / "status.json", {"identity": "old"})
        write_json(
            new / "verification.json",
            {"valid": True, "identity": "new", "molecules": 400},
        )
        write_json(new / "runtime_migration.json", {"old_release": str(old)})
        controller.register_round("round", "generation_model", [], root / "report.json")
        controller.mark_round("round", status="eligible", release=str(old))
        return controller, old, new

    def test_ambiguous_or_accepted_submission_cannot_be_replaced(self):
        for previous in (
            {"id": 123, "status": "COMPLETE"},
            {"id": None, "status": "intent"},
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                controller, old, new = self.setup_case(root)
                controller.change(
                    lambda s, previous=previous: s["submissions"].update(old=previous)
                )
                with (
                    patch(
                        "casmi_ml.runtime_migration.Controller", return_value=controller
                    ),
                    self.assertRaises(ValueError),
                ):
                    activate_round("round", new)
                row = controller.read()["rounds"][0]
                self.assertEqual(row["release"], str(old))
                self.assertEqual(controller.read()["submissions"]["old"], previous)

    def test_mismatched_full_ranking_blocks_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller, old, new = self.setup_case(root)
            frame = pd.read_csv(new / "local_output/submission.csv")
            frame.loc[399, "rank"] = "different"
            frame.to_csv(new / "local_output/submission.csv", index=False)
            with (
                patch("casmi_ml.runtime_migration.Controller", return_value=controller),
                self.assertRaises(ValueError),
            ):
                activate_round("round", new)
            self.assertEqual(controller.read()["rounds"][0]["release"], str(old))
