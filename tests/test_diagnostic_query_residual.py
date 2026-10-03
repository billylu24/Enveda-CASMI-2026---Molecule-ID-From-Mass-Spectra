import unittest

import torch

from casmi_ml.diagnostic_query_residual import DiagnosticQueryResidual
from casmi_ml.diagnostic_residual import DIM


class QueryResidualTests(unittest.TestCase):
    def test_zero_initialization_and_empty_evidence_preserve_original_scores_exactly(
        self,
    ):
        torch.manual_seed(42)
        model = DiagnosticQueryResidual()
        query = torch.nn.functional.normalize(torch.randn(3, 128), dim=-1)
        latent = torch.randn(3, 768)
        chemical = torch.ones(3, DIM)
        structures = torch.nn.functional.normalize(torch.randn(15, 128), dim=-1)
        output = model(query, latent, chemical)
        torch.testing.assert_close(output, query, rtol=0, atol=0)
        torch.testing.assert_close(
            output @ structures.T, query @ structures.T, rtol=0, atol=0
        )
        model.correction[-1].bias.data.fill_(0.5)
        torch.testing.assert_close(
            model(query, latent, torch.zeros_like(chemical)), query, rtol=0, atol=0
        )

    def test_zero_initialized_residual_receives_training_gradient(self):
        torch.manual_seed(42)
        model = DiagnosticQueryResidual()
        query = torch.nn.functional.normalize(torch.randn(3, 128), dim=-1)
        latent = torch.randn(3, 768)
        chemical = torch.ones(3, DIM)
        structure = torch.nn.functional.normalize(torch.randn(3, 128), dim=-1)
        loss = -(model(query, latent, chemical) * structure).sum()
        loss.backward()
        self.assertGreater(float(model.correction[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.correction[-1].bias.grad.abs().sum()), 0)

    def test_matched_mask_removes_identity_but_preserves_presence_gate(self):
        torch.manual_seed(42)
        model = DiagnosticQueryResidual(mask_chemical_channels=True)
        torch.nn.init.normal_(model.correction[-1].weight)
        query = torch.nn.functional.normalize(torch.randn(1, 128), dim=-1)
        latent = torch.randn(1, 768)
        first = torch.zeros(1, DIM)
        first[:, 1] = 1
        second = torch.zeros(1, DIM)
        second[:, 5] = 1
        torch.testing.assert_close(
            model(query, latent, first), model(query, latent, second), rtol=0, atol=0
        )
        torch.testing.assert_close(
            model(query, latent, torch.zeros_like(first)), query, rtol=0, atol=0
        )
