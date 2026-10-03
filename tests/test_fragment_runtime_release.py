import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.fragment_runtime_release import activate
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity


class FragmentRuntimeActivationTests(unittest.TestCase):
    def fixture(self, root):
        import pandas as pd

        config = json.loads(Path("configs/research_loop.json").read_text())
        config["root"] = str(root / "state")
        write_json(root / "config.json", config)
        c = Controller(root / "config.json")
        old, new = root / "old", root / "new"
        decision = {
            "direction": "chembl_high_fragment",
            "winner": {"variant": "high_relative_fragment", "gate": {"eligible": True}},
        }
        for release in (old, new):
            (release / "bundle").mkdir(parents=True)
            (release / "bundle/model.pt").write_bytes(b"frozen")
            write_json(release / "bundle/development_decision.json", decision)
            recipe = {"external_routed": {"high_fragment": {"prefix": 3}}}
            if release == new:
                recipe["external_routed"]["high_fragment"]["runtime"] = {
                    "candidate_threads": 2,
                    "polling_ms": 10,
                }
            write_json(release / "bundle/deployment_recipe.json", recipe)
            sums = {
                str(p.relative_to(release / "bundle")): digest(p)
                for p in (release / "bundle").glob("*")
            }
            write_json(release / "bundle/SHA256SUMS.json", sums)
            write_json(
                release / "notebook/kernel-metadata.json",
                {"id": "owner/" + release.name},
            )
            write_json(release / "notebook/code.ipynb", {"cells": []})
            identity = release_identity(release, sums, "high_relative_fragment")
            write_json(release / "status.json", {"identity": identity})
            write_json(
                release / "verification.json",
                {
                    "identity": identity,
                    "valid": True,
                    "molecules": 400,
                    "seconds": 100,
                    "peak_rss_mib": 100,
                },
            )
            (release / "local_output").mkdir()
            pd.DataFrame({"molecule_id": range(400), "smiles": ["CCO"] * 400}).to_csv(
                release / "local_output/submission.csv", index=False
            )
        write_json(new / "runtime_migration.json", {"old_release": str(old)})
        write_json(
            new / "implementation_replay.json",
            {
                "valid": True,
                "full_rank_matches": 75,
                "fresh_fragment_cache": True,
                "bundle_sums_sha256": digest(new / "bundle/SHA256SUMS.json"),
            },
        )
        c.register_round("high", "chembl_high_fragment", [], root / "report.json")
        c.mark_round("high", status="eligible", release=str(old))
        return c, old, new

    def test_incomplete_replay_or_resource_failure_cannot_activate(self):
        for fields in ({"full_rank_matches": 74}, {"fresh_fragment_cache": False}):
            with tempfile.TemporaryDirectory() as directory:
                c, old, new = self.fixture(Path(directory))
                replay = json.loads((new / "implementation_replay.json").read_text())
                replay.update(fields)
                write_json(new / "implementation_replay.json", replay)
                with (
                    patch("casmi_ml.research_loop.Controller", return_value=c),
                    self.assertRaisesRegex(ValueError, "runtime/resource"),
                ):
                    activate("high", new)
                self.assertEqual(c.read()["rounds"][0]["release"], str(old))
        with tempfile.TemporaryDirectory() as directory:
            c, old, new = self.fixture(Path(directory))
            v = json.loads((new / "verification.json").read_text())
            v["seconds"] = 1801
            write_json(new / "verification.json", v)
            with (
                patch("casmi_ml.research_loop.Controller", return_value=c),
                self.assertRaisesRegex(ValueError, "runtime/resource"),
            ):
                activate("high", new)
            self.assertEqual(c.read()["rounds"][0]["release"], str(old))

    def test_any_changed_top25_cannot_activate(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            c, old, new = self.fixture(Path(directory))
            csv = new / "local_output/submission.csv"
            data = pd.read_csv(csv)
            data.loc[399, "smiles"] = "CCN"
            data.to_csv(csv, index=False)
            with (
                patch("casmi_ml.research_loop.Controller", return_value=c),
                self.assertRaisesRegex(ValueError, "All400"),
            ):
                activate("high", new)
            self.assertEqual(c.read()["rounds"][0]["release"], str(old))

    def test_accepted_or_ambiguous_submission_cannot_activate(self):
        for status, identifier in (("PENDING", 123), ("intent", None)):
            with tempfile.TemporaryDirectory() as directory:
                c, old, new = self.fixture(Path(directory))
                identity = json.loads((old / "status.json").read_text())["identity"]
                c.change(
                    lambda s, identity=identity, status=status, identifier=identifier: (
                        s["submissions"].update(
                            {identity: {"status": status, "id": identifier}}
                        )
                    )
                )
                with (
                    patch("casmi_ml.research_loop.Controller", return_value=c),
                    self.assertRaisesRegex(ValueError, "accepted or ambiguous"),
                ):
                    activate("high", new)
                self.assertEqual(c.read()["rounds"][0]["release"], str(old))

    def test_unchanged_runtime_can_activate_but_still_requires_platform_verification(
        self,
    ):
        from contextlib import chdir

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c, old, new = self.fixture(root)
            write_json(root / "configs/research_publish_paths.json", {"paths": []})
            with (
                chdir(root),
                patch("casmi_ml.research_loop.Controller", return_value=c),
            ):
                amendment = activate("high", new)
            entry = c.read()["rounds"][0]
            self.assertEqual(entry["release"], str(new))
            self.assertTrue(entry["requires_platform_verification"])
            self.assertIsNone(entry["remote_release"])
            self.assertFalse(entry["dataset_uploaded"])
            self.assertIsNone(c.read()["pending_submission"])
            self.assertEqual(amendment["old_release"], str(old))
