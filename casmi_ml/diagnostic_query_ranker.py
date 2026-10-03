"""Frozen original candidate critic with a chemical query residual."""

import numpy as np
import torch
from torch import nn

from casmi_ml.diagnostic_query_residual import DiagnosticQueryResidual
from casmi_ml.diagnostic_residual import evidence_features
from casmi_ml.direct_models import DirectRanker
from casmi_ml.metfrag import digest


class DiagnosticQueryRanker(nn.Module):
    uses_chemical_query = True

    def __init__(self, base, mask_chemical_channels=False):
        super().__init__()
        if base.architecture != "fingerprint":
            raise ValueError("Frozen fingerprint critic required")
        self.base = base.requires_grad_(False).eval()
        self.architecture = base.architecture
        self.query_residual = DiagnosticQueryResidual(mask_chemical_channels)

    @property
    def log_scale(self):
        return self.base.log_scale

    def encode_molecules(self, fps, graphs=None):
        return self.base.encode_molecules(fps, graphs)

    def encode_spectra(self, latent, chemical=None):
        if chemical is None:
            raise ValueError("Observable precise chemical evidence required")
        return self.query_residual(self.base.encode_spectra(latent), latent, chemical)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self


def load_query_ranker(path, original):
    original, path = str(original), str(path)
    base_weights = torch.load(original, map_location="cpu", weights_only=True)
    weights = torch.load(path, map_location="cpu", weights_only=True)
    meta = weights.get("diagnostic_query_residual", {})
    if (
        meta.get("base_critic_sha256") != digest(original)
        or meta.get("dimension") != 24
        or not meta.get("precise")
    ):
        raise ValueError("Query residual original critic/feature binding differs")
    base = DirectRanker("fingerprint")
    base.load_state_dict(base_weights["state_dict"])
    model = DiagnosticQueryRanker(base, meta.get("mask_chemical_channels", False))
    state = weights["state_dict"]
    expected = {"base." + key for key in base_weights["state_dict"]}
    if {key for key in state if key.startswith("base.")} != expected:
        raise ValueError("Original query critic tensor keys differ")
    for key, value in base_weights["state_dict"].items():
        if not torch.equal(value, state["base." + key]):
            raise ValueError("Original query critic tensor changed: " + key)
    if weights["encoder_sha256"] != base_weights["encoder_sha256"]:
        raise ValueError("Query critic encoder binding differs")
    model.load_state_dict(state)
    return model.eval(), weights


@torch.inference_mode()
def score_chemical_group(ranker, encoder, group, preprocessing, candidates, fps):
    from casmi_ml.data import features

    rows = [features(row, preprocessing) for row in group.to_dict("records")]
    pieces = [torch.from_numpy(np.stack([r[i] for r in rows])) for i in [0, 2, 1]]
    latent = encoder.encoder(torch.cat(pieces, -1))
    chemical = torch.from_numpy(
        np.stack([evidence_features(row, True) for row in group.to_dict("records")])
    )
    query = ranker.encode_spectra(latent, chemical).mean(0)
    return (
        np.concatenate(
            [
                (
                    ranker.encode_molecules(torch.from_numpy(fps[start : start + 256]))
                    @ query
                ).numpy()
                for start in range(0, len(candidates), 256)
            ]
        )
        if len(candidates)
        else np.empty(0, np.float32)
    )
