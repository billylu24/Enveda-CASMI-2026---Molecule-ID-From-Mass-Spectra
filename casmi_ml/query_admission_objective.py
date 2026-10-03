"""Tie training positive logits to the frozen deployed cosine admission margin."""

import torch

ADMISSION_MARGIN = 0.05


def admission_logits(cosine_scores, scale):
    if cosine_scores.ndim != 2 or cosine_scores.shape[1] != 16:
        raise ValueError("Positive plus15 pinned cosine scores required")
    if not bool(torch.isfinite(cosine_scores).all()):
        raise ValueError("Finite cosine scores required")
    adjusted = cosine_scores.clone()
    adjusted[:, 0] -= ADMISSION_MARGIN
    return adjusted * scale
