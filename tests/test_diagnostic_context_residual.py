import tempfile
import unittest
from pathlib import Path

import torch

from casmi_ml.diagnostic_residual import DIM, DiagnosticResidual
from casmi_ml.scale_models import ScaleModel
from casmi_ml.secondary_inference import load_deployment_checkpoint


class ContextualDiagnosticTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        base = ScaleModel("wide_enhanced", 7).eval()
        model = DiagnosticResidual(base, contextual=True).eval()
        batch = {
            "hist": torch.randn(2, 1250),
            "loss": torch.randn(2, 1250),
            "meta": torch.randn(2, 7),
            "chemical": torch.ones(2, DIM),
        }
        return base, model, batch

    def test_zero_initialization_and_no_evidence_preserve_base_and_latent_exact(self):
        base, model, batch = self.fixture()
        with torch.inference_mode():
            torch.testing.assert_close(model(batch), base(batch), rtol=0, atol=0)
            model.correction[-1].bias.fill_(1)
            batch["chemical"].zero_()
            torch.testing.assert_close(model(batch), base(batch), rtol=0, atol=0)
        self.assertIs(model.encoder, base.encoder)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))

    def test_contextual_checkpoint_roundtrip_uses_matching_input_dimensions(self):
        _, model, batch = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            torch.save(
                {
                    "architecture": "wide_enhanced",
                    "metadata_dim": 7,
                    "state_dict": model.state_dict(),
                    "diagnostic_residual": {"contextual": True},
                },
                path,
            )
            loaded, _ = load_deployment_checkpoint(path, "scale")
            with torch.inference_mode():
                torch.testing.assert_close(loaded(batch), model(batch), rtol=0, atol=0)
            self.assertEqual(loaded.correction[0].in_features, DIM + 768)

    def test_mask_control_retains_strength_gate_and_removes_chemical_identity(self):
        base, _, _ = self.fixture()
        model = DiagnosticResidual(base, contextual=True, mask_chemical_channels=True)
        model.correction[-1].bias.data.fill_(1)
        context = torch.ones(2, 768)
        chemical = torch.zeros(2, DIM)
        chemical[:, 0] = 1
        first = model.residual(chemical, context)
        chemical[:, 0] = 0
        chemical[:, 5] = 1
        torch.testing.assert_close(
            model.residual(chemical, context), first, rtol=0, atol=0
        )
        chemical.zero_()
        torch.testing.assert_close(
            model.residual(chemical, context), torch.zeros_like(first), rtol=0, atol=0
        )

    def test_nonzero_contextual_correction_requires_original_latent(self):
        _, model, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "frozen768"):
            model.residual(torch.ones(2, DIM))


class PeakLossAttributionTests(unittest.TestCase):
    def test_mask_removes_peak_loss_identity_while_retaining_strength_gate(self):
        torch.manual_seed(42)
        base = ScaleModel("wide_enhanced", 7).eval()
        model = DiagnosticResidual(base, contextual=True, mask_peak_losses=True).eval()
        torch.nn.init.normal_(model.correction[-1].weight)
        context = torch.ones(1, 768)
        first = torch.zeros(1, DIM)
        first[:, 2] = 1
        second = torch.zeros(1, DIM)
        second[:, 5] = 1
        torch.testing.assert_close(
            model.residual(first, context),
            model.residual(second, context),
            rtol=0,
            atol=0,
        )
        precursor = torch.zeros(1, DIM)
        precursor[:, 1] = 1
        self.assertFalse(
            torch.equal(
                model.residual(first, context), model.residual(precursor, context)
            )
        )
        torch.testing.assert_close(
            model.residual(torch.zeros_like(first), context),
            torch.zeros(1, 2048),
            rtol=0,
            atol=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {
                    "architecture": "wide_enhanced",
                    "metadata_dim": 7,
                    "state_dict": model.state_dict(),
                    "diagnostic_residual": {
                        "contextual": True,
                        "mask_peak_losses": True,
                    },
                },
                path,
            )
            loaded, _ = load_deployment_checkpoint(path, "scale")
            torch.testing.assert_close(
                loaded.residual(first, context),
                model.residual(first, context),
                rtol=0,
                atol=0,
            )
