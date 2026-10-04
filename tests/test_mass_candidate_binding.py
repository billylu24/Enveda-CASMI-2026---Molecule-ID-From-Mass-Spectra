import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.mass_candidate_compatibility import load_model


class MassCandidateBindingTest(unittest.TestCase):
    def write_model(self, root, **fields):
        path = Path(root) / "model.json"
        path.write_text(json.dumps(fields))
        return path

    def test_altered_training_mask_binding_rejected_before_original_loader(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_model(
                root,
                mass_admissible_denominator=True,
                mass_mask_source_sha256="changed",
            )
            with (
                patch(
                    "casmi_ml.mass_candidate_compatibility.bindings",
                    return_value={"mass_mask_source_sha256": "original"},
                ),
                patch("casmi_ml.mass_candidate_compatibility.original_load") as loader,
            ):
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    load_model(path, "encoder", "critic")
                loader.assert_not_called()

    def test_valid_training_provenance_still_requires_original_inference_bindings(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_model(
                root,
                mass_admissible_denominator=True,
                mass_mask_source_sha256="original",
            )
            with (
                patch(
                    "casmi_ml.mass_candidate_compatibility.bindings",
                    return_value={"mass_mask_source_sha256": "original"},
                ),
                patch(
                    "casmi_ml.mass_candidate_compatibility.original_load",
                    side_effect=ValueError("Original encoder binding differs"),
                ) as loader,
            ):
                with self.assertRaisesRegex(ValueError, "encoder binding"):
                    load_model(path, "encoder", "critic")
                loader.assert_called_once_with(path, "encoder", "critic")

    def test_unmarked_old_model_cannot_enter_new_system(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_model(root, weights=[0] * 60)
            with self.assertRaisesRegex(ValueError, "provenance required"):
                load_model(path, "encoder", "critic")
