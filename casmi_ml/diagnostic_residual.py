"""Observable exact chemical evidence as a frozen-encoder fingerprint residual."""

import numpy as np
import torch
from torch import nn

from casmi_ml.chemistry import RULES, extract_evidence

KINDS = ("ion", "precursor_loss", "peak_loss")
DIM = len(RULES) * len(KINDS)


def legacy_evidence_features(row, precise=True):
    result = np.zeros(DIM, np.float32)
    names = {rule.name: i for i, rule in enumerate(RULES)}
    for match in extract_evidence(
        row, ppm=10, absolute_da=0.002 if precise else 0.5, include_combinations=False
    )["matches"]:
        kind = match["kind"]
        if kind in KINDS:
            i = names[match["rule"]] * len(KINDS) + KINDS.index(kind)
            result[i] = max(result[i], match["strength"])
    return result


def evidence_features(row, precise=True):
    from casmi_ml.diagnostic_feature_runtime import maximum_evidence_features

    return maximum_evidence_features(row, precise)


class DiagnosticResidual(nn.Module):
    def __init__(self, base, contextual=False, mask_chemical_channels=False):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.contextual = contextual
        self.mask_chemical_channels = mask_chemical_channels
        if mask_chemical_channels and not contextual:
            raise ValueError(
                "Chemical-channel attribution mask requires contextual residual"
            )
        if contextual and getattr(base, "architecture", None) != "wide_enhanced":
            raise ValueError(
                "Contextual residual requires the frozen wide_enhanced base"
            )
        self.correction = nn.Sequential(
            nn.Linear(DIM + (768 if contextual else 0), 64),
            nn.GELU(),
            nn.Linear(64, 2048),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    @property
    def encoder(self):
        return self.base.encoder

    def residual(self, chemical, context=None):
        # No evidence means exactly no alteration to the original encoder.
        features = (
            torch.zeros_like(chemical) if self.mask_chemical_channels else chemical
        )
        if self.contextual:
            if context is None or context.shape[-1] != 768:
                raise ValueError(
                    "Contextual evidence correction needs frozen768 latent"
                )
            features = torch.cat([features, context], -1)
        return self.correction(features) * chemical.amax(-1, keepdim=True)

    def forward(self, batch):
        if self.contextual:
            latent = self.base.encoder(
                torch.cat([batch["hist"], batch["meta"], batch["loss"]], -1)
            )
            return self.base.head(latent) + self.residual(batch["chemical"], latent)
        return self.base(batch) + self.residual(batch["chemical"])

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self
