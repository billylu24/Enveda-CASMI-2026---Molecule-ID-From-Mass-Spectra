import tempfile
import unittest
from pathlib import Path

import torch

from casmi_ml.diagnostic_system_evaluation import validate_residual_binding
from casmi_ml.metfrag import digest


class ResidualCriticBindingTests(unittest.TestCase):
    def fixture(self, directory):
        original, residual = directory / "base.pt", directory / "residual.pt"
        base = {
            "architecture": "wide_enhanced",
            "metadata_dim": 7,
            "preprocessing": {"seed": 42},
            "state_dict": {"encoder.weight": torch.ones(2, 3)},
        }
        torch.save(base, original)
        wrapped = dict(
            base,
            state_dict={"base.encoder.weight": torch.ones(2, 3)},
            preprocessing={"seed": 42, "diagnostic_precision": True},
            diagnostic_residual={
                "base_sha256": digest(original),
                "dimension": 24,
                "precise": True,
            },
        )
        torch.save(wrapped, residual)
        return original, residual, wrapped

    def test_same_base_is_allowed_and_a_single_changed_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            original, residual, wrapped = self.fixture(Path(d))
            self.assertTrue(
                validate_residual_binding(residual, original)["all_base_tensors_exact"]
            )
            wrapped["state_dict"]["base.encoder.weight"][0, 0] += 0.01
            torch.save(wrapped, residual)
            with self.assertRaisesRegex(ValueError, "changed frozen base"):
                validate_residual_binding(residual, original)

    def test_preprocessing_change_is_rejected_even_if_base_tensors_match(self):
        with tempfile.TemporaryDirectory() as d:
            original, residual, wrapped = self.fixture(Path(d))
            wrapped["preprocessing"]["seed"] = 3
            torch.save(wrapped, residual)
            with self.assertRaisesRegex(ValueError, "preprocessing differs"):
                validate_residual_binding(residual, original)
