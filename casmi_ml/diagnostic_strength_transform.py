"""Prespecified weak peak-loss attenuation, leaving original evidence gate available."""

import torch

PEAK_LOSS_FLOOR = 0.3


def strong_peak_loss_features(chemical):
    if chemical.shape[-1] != 24:
        raise ValueError("Expected24 rule/kind chemistry channels")
    result = chemical.clone()
    result[..., 2::3] = torch.relu(chemical[..., 2::3] - PEAK_LOSS_FLOOR) / (
        1 - PEAK_LOSS_FLOOR
    )
    return result
