import unittest

import torch
from torch.nn import functional as F

from casmi_ml.chemical_query_constraint import (
    constrain_context_weights,
    verify_context_weights,
)
from casmi_ml.diagnostic_query_residual import DiagnosticQueryResidual


class ChemicalQueryConstraintTests(unittest.TestCase):
    def test_adam_weight_decay_preserves_zero_context_and_learns_identity(self):
        torch.manual_seed(42)
        model = DiagnosticQueryResidual()
        constrain_context_weights(model)
        query = F.normalize(torch.randn(8, 128), dim=-1)
        latent = torch.randn(8, 768)
        chemical = torch.rand(8, 24)
        target = F.normalize(torch.randn(8, 128), dim=-1)
        torch.testing.assert_close(
            model(query, latent, chemical), query, rtol=0, atol=0
        )
        before = model.correction[0].weight[:, 768:].detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = -(model(query, latent, chemical) * target).sum()
            loss.backward()
            self.assertEqual(
                int(model.correction[0].weight.grad[:, :768].count_nonzero()), 0
            )
            optimizer.step()
            verify_context_weights(model)
        self.assertFalse(torch.equal(before, model.correction[0].weight[:, 768:]))
        torch.testing.assert_close(
            model(query, latent, chemical),
            model(query, latent * 7, chemical),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model(query, latent, torch.zeros_like(chemical)), query, rtol=0, atol=0
        )

    def test_matched_mask_keeps_gate_but_removes_chemical_identity(self):
        torch.manual_seed(42)
        model = DiagnosticQueryResidual(mask_chemical_channels=True)
        constrain_context_weights(model)
        torch.nn.init.normal_(model.correction[-1].weight)
        query = F.normalize(torch.randn(3, 128), dim=-1)
        latent = torch.randn(3, 768)
        a = torch.zeros(3, 24)
        a[:, 0] = 1
        b = torch.zeros(3, 24)
        b[:, 5] = 1
        torch.testing.assert_close(
            model(query, latent, a), model(query, latent * 5, b), rtol=0, atol=0
        )
        model.correction[0].weight.data[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "nonzero latent"):
            verify_context_weights(model)
