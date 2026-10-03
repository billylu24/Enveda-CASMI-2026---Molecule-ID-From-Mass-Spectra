"""Observable exact chemical evidence as a frozen-encoder fingerprint residual."""

import numpy as np
from torch import nn

from casmi_ml.chemistry import RULES, extract_evidence

KINDS = ("ion", "precursor_loss", "peak_loss")
DIM = len(RULES) * len(KINDS)


def evidence_features(row, precise=True):
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


class DiagnosticResidual(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.correction = nn.Sequential(
            nn.Linear(DIM, 64), nn.GELU(), nn.Linear(64, 2048)
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    @property
    def encoder(self):
        return self.base.encoder

    def residual(self, chemical):
        # No evidence means exactly no alteration to the original encoder.
        return self.correction(chemical) * chemical.amax(-1, keepdim=True)

    def forward(self, batch):
        return self.base(batch) + self.residual(batch["chemical"])

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self
