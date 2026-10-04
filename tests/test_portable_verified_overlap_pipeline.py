import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml import portable_verified_overlap_pipeline as runtime


class VerifiedOverlapBindingTest(unittest.TestCase):
    def test_changed_source_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            recipe = p / "recipe.json"
            recipe.write_text("{}")
            proof = {
                "sources": {n: "bound" for n in runtime.SOURCES},
                "full_rank_matches": {"unknown": 2000, "known": 1340},
                "all400_original_generation_handoffs_and_events_exact": True,
                "original_generation_source_sha256": "bound",
            }
            (p / "release_evidence.json").write_text(
                json.dumps({"overlap_runtime": proof})
            )
            with (
                patch.object(runtime, "digest", return_value="changed"),
                patch.object(runtime, "overlap_run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    runtime.run(recipe, "data", "catalog", "output", True)
                run.assert_not_called()

    def test_one_gpu_falls_back_to_frozen_balanced_pipeline_unless_forced(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            recipe = p / "recipe.json"
            recipe.write_text("{}")
            proof = {
                "sources": {n: "bound" for n in runtime.SOURCES},
                "full_rank_matches": {"unknown": 2000, "known": 1340},
                "all400_original_generation_handoffs_and_events_exact": True,
                "original_generation_source_sha256": "bound",
            }
            (p / "release_evidence.json").write_text(
                json.dumps({"overlap_runtime": proof})
            )
            with (
                patch.object(runtime, "digest", return_value="bound"),
                patch.object(runtime.torch.cuda, "device_count", return_value=1),
                patch(
                    "casmi_ml.portable_balanced_cpu_pipeline.run",
                    return_value="balanced",
                ) as balanced,
                patch.object(runtime, "overlap_run", return_value="overlap") as overlap,
            ):
                self.assertEqual(
                    runtime.run(recipe, "data", "catalog", "output"), "balanced"
                )
                balanced.assert_called_once()
                overlap.assert_not_called()
                self.assertEqual(
                    runtime.run(recipe, "data", "catalog", "output", True), "overlap"
                )
                overlap.assert_called_once_with(
                    recipe, "data", "catalog", "output", True
                )


if __name__ == "__main__":
    unittest.main()
