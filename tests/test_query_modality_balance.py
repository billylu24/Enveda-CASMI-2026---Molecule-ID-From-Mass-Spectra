import unittest

import torch
from torch.nn import functional as F

from casmi_ml.diagnostic_query_residual import DiagnosticQueryResidual
from casmi_ml.query_modality_balance import balanced_query, fit_scales, fold_scales


class QueryModalityBalanceTests(unittest.TestCase):
    def test_train_only_scales_balance_total_energy_and_fold_into_existing_graph(self):
        torch.manual_seed(42)
        latent = torch.randn(11, 2, 768) * 9
        chemical = torch.rand(11, 2, 24) * 0.05
        scales = fit_scales(latent, chemical)
        for value, name in [(latent, "latent"), (chemical, "chemical")]:
            self.assertAlmostEqual(
                float((value * scales[name]).square().sum(-1).mean()), 1, places=6
            )
        query = F.normalize(torch.randn(11, 2, 128), dim=-1)
        for mask in [False, True]:
            model = DiagnosticQueryResidual(mask_chemical_channels=mask)
            torch.testing.assert_close(
                balanced_query(model, query, latent, chemical, scales),
                query,
                rtol=0,
                atol=0,
            )
            torch.nn.init.normal_(model.correction[-1].weight, std=0.05)
            expected = balanced_query(model, query, latent, chemical, scales)
            fold_scales(model, scales)
            torch.testing.assert_close(
                model(query, latent, chemical), expected, rtol=1e-6, atol=1e-7
            )
            torch.testing.assert_close(
                model(query, latent, torch.zeros_like(chemical)), query, rtol=0, atol=0
            )

    def test_invalid_and_zero_energy_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "Aligned"):
            fit_scales(torch.ones(2, 768), torch.ones(3, 24))
        for chemical in [torch.zeros(2, 24), torch.full((2, 24), float("nan"))]:
            with self.assertRaisesRegex(ValueError, "energies"):
                fit_scales(torch.ones(2, 768), chemical)
