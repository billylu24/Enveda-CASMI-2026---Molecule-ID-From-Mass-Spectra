import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml import portable_verified_overlap_cpu_pipeline as runtime


class VerifiedCPUOverlapBindingTest(unittest.TestCase):
    def recipe(self, root):
        recipe = Path(root) / "recipe.json"
        recipe.write_text("{}")
        proof = {
            "sources": {name: "bound" for name in runtime.SOURCES},
            "full_rank_matches": {"unknown": 2000, "known": 1340},
            "all400_generation_events_exact": True,
            "all400_generation_csv_audit_fullhandoffs_exact": True,
            "all400_top25_exact": True,
            "all265_raw_physical_dictionaries_exact": True,
            "torch_cpu_threads": 1,
        }
        (Path(root) / "release_evidence.json").write_text(
            json.dumps({"generation_cpu_runtime": proof})
        )
        return recipe, proof

    def test_changed_worker_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as root:
            recipe, _ = self.recipe(root)
            with (
                patch.object(runtime, "digest", return_value="changed"),
                patch.object(runtime, "cpu_run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    runtime.run(recipe, "data", "catalog", "output", True)
                run.assert_not_called()

    def test_incomplete_physical_identity_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as root:
            recipe, proof = self.recipe(root)
            proof["all265_raw_physical_dictionaries_exact"] = False
            (Path(root) / "release_evidence.json").write_text(
                json.dumps({"generation_cpu_runtime": proof})
            )
            with patch.object(runtime, "cpu_run") as run:
                with self.assertRaisesRegex(ValueError, "controls required"):
                    runtime.run(recipe, "data", "catalog", "output", True)
                run.assert_not_called()

    def test_valid_source_passes_original_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            recipe, _ = self.recipe(root)
            with (
                patch.object(runtime, "digest", return_value="bound"),
                patch.object(runtime, "cpu_run", return_value="original result") as run,
            ):
                self.assertEqual(
                    runtime.run(recipe, "data", "catalog", "output", True),
                    "original result",
                )
                run.assert_called_once_with(recipe, "data", "catalog", "output", True)
