import unittest

import torch
from torch.nn import functional as F

from casmi_ml.diagnostic_query_residual import DiagnosticQueryResidual


class StrongPeakQueryCompatibilityTests(unittest.TestCase):
    def test_optional_strong_zero_and_empty_evidence_preserve_baseline_exact(self):
        torch.manual_seed(42)
        query = F.normalize(torch.randn(7, 128), dim=-1)
        latent = torch.randn(7, 768)
        chemical = torch.rand(7, 24)
        strong = DiagnosticQueryResidual(strong_peak_losses=True)
        torch.testing.assert_close(
            strong(query, latent, chemical), query, rtol=0, atol=0
        )
        torch.nn.init.normal_(strong.correction[-1].weight)
        torch.testing.assert_close(
            strong(query, latent, torch.zeros_like(chemical)), query, rtol=0, atol=0
        )

    def test_identity_mask_keeps_original_raw_gate(self):
        torch.manual_seed(42)
        model = DiagnosticQueryResidual(
            mask_chemical_channels=True, strong_peak_losses=True
        )
        torch.nn.init.normal_(model.correction[-1].weight)
        query = F.normalize(torch.randn(3, 128), dim=-1)
        latent = torch.randn(3, 768)
        peak = torch.zeros(3, 24)
        peak[:, 2] = 0.2
        ion = torch.zeros(3, 24)
        ion[:, 21] = 0.2
        torch.testing.assert_close(
            model(query, latent, peak), model(query, latent, ion), rtol=0, atol=0
        )
        self.assertFalse(torch.equal(model(query, latent, peak), query))
