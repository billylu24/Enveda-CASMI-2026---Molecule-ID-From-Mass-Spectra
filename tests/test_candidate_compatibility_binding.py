import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.candidate_diagnostic_compatibility import load_model


class CandidateBindingTests(unittest.TestCase):
    def model(self):
        return {
            "weights": [0.0] * 24,
            "bonus_budget": 0.05,
            "mask_chemical_channels": False,
            "local_only": True,
            **{
                name: "bound"
                for name in [
                    "encoder_sha256",
                    "original_critic_sha256",
                    "compatibility_source_sha256",
                    "chemistry_source_sha256",
                    "evidence_source_sha256",
                    "feature_source_sha256",
                    "training_sha256",
                ]
            },
        }

    def test_all_input_and_feature_sources_are_bound(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("casmi_ml.metfrag.digest", return_value="bound"),
        ):
            path = Path(tmp) / "model.json"
            model = self.model()
            path.write_text(json.dumps(model))
            score, loaded = load_model(path, "encoder", "critic")
            self.assertTrue(callable(score))
            self.assertEqual(model, loaded)
            for binding in [k for k in model if k.endswith("_sha256")]:
                changed = dict(model, **{binding: "changed"})
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, binding):
                    load_model(path, "encoder", "critic")

    def test_loader_rejects_budget_change_or_nonlocal_model(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("casmi_ml.metfrag.digest", return_value="bound"),
        ):
            path = Path(tmp) / "model.json"
            for changed in [
                dict(self.model(), bonus_budget=0.1),
                dict(self.model(), local_only=False),
            ]:
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "budget"):
                    load_model(path, "encoder", "critic")
