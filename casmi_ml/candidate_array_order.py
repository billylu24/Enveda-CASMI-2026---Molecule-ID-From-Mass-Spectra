"""Exact scalar score/key ordering without per-candidate pandas row access."""

import numpy as np


def rank_values(keys, values):
    values = np.asarray(values)
    if values.ndim != 1 or len(keys) != len(values) or not np.isfinite(values).all():
        raise ValueError(
            "Finite one-dimensional candidate scores and matching keys required"
        )
    if len(set(keys)) != len(keys) or any(not isinstance(key, str) for key in keys):
        raise ValueError("Unique string candidate keys required")
    order = np.lexsort((np.asarray(keys), -values))
    return [keys[i] for i in order]
