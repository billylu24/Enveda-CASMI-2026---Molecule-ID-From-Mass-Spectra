"""Train-only scalar modality normalization folded into the verified query graph."""

import torch
from torch.nn import functional as F


def fit_scales(latent, chemical):
    if (
        latent.shape[-1] != 768
        or chemical.shape[-1] != 24
        or latent.shape[:-1] != chemical.shape[:-1]
    ):
        raise ValueError("Aligned latent768 and chemical24 required")
    context_energy = latent.double().square().sum(-1).mean()
    chemical_energy = chemical.double().square().sum(-1).mean()
    if (
        not torch.isfinite(context_energy + chemical_energy)
        or min(float(context_energy), float(chemical_energy)) <= 0
    ):
        raise ValueError("Finite nonzero training modality energies required")
    # Each input block has average squared L2 energy one; a single scalar per
    # block preserves relative channels and avoids amplifying rare dimensions.
    return {
        "latent": float(context_energy.rsqrt()),
        "chemical": float(chemical_energy.rsqrt()),
    }


def balanced_query(residual, baseline, latent, chemical, scales):
    identity = (
        torch.zeros_like(chemical) if residual.mask_chemical_channels else chemical
    )
    delta = residual.correction(
        torch.cat([latent * scales["latent"], identity * scales["chemical"]], -1)
    ) * chemical.amax(-1, keepdim=True)
    return baseline + (
        F.normalize(baseline + delta, dim=-1) - F.normalize(baseline, dim=-1)
    )


def fold_scales(residual, scales):
    if any(not (0 < value < float("inf")) for value in scales.values()):
        raise ValueError("Positive finite modality scales required")
    with torch.no_grad():
        residual.correction[0].weight[:, :768].mul_(scales["latent"])
        residual.correction[0].weight[:, 768:].mul_(scales["chemical"])
