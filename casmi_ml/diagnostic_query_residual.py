"""Candidate-facing chemical residual over a frozen normalized critic query."""

import torch
from torch import nn
from torch.nn import functional as F

from casmi_ml.diagnostic_residual import DIM


class DiagnosticQueryResidual(nn.Module):
    def __init__(self, mask_chemical_channels=False):
        super().__init__()
        self.mask_chemical_channels = mask_chemical_channels
        self.correction = nn.Sequential(
            nn.Linear(768 + DIM, 64), nn.GELU(), nn.Linear(64, 128)
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, baseline_query, latent, chemical):
        if (
            baseline_query.shape[-1] != 128
            or latent.shape[-1] != 768
            or chemical.shape[-1] != DIM
        ):
            raise ValueError(
                "Frozen critic query128, encoder latent768 and chemical24 required"
            )
        features = (
            torch.zeros_like(chemical) if self.mask_chemical_channels else chemical
        )
        delta = self.correction(torch.cat([latent, features], -1)) * chemical.amax(
            -1, keepdim=True
        )
        # Subtract the identical normalized baseline in the residual expression.
        # At zero initialization and without evidence, the difference is exactly
        # zero; avoid normalizing an already-normalized baseline a second time.
        change = F.normalize(baseline_query + delta, dim=-1) - F.normalize(
            baseline_query, dim=-1
        )
        return baseline_query + change
