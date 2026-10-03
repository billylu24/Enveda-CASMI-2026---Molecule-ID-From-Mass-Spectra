"""Training-only admissibility of pinned negatives under observed mass windows."""

import numpy as np


def admissible_negatives(masses, hard):
    masses = np.asarray(masses, dtype=np.float64)
    hard = np.asarray(hard)
    if masses.ndim != 1 or not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("Finite positive training masses required")
    if (
        hard.ndim != 2
        or len(hard) != len(masses)
        or not np.issubdtype(hard.dtype, np.integer)
    ):
        raise ValueError("Pinned training hard-negative matrix required")
    if (
        (hard < 0).any()
        or (hard >= len(masses)).any()
        or (hard == np.arange(len(masses))[:, None]).any()
    ):
        raise ValueError("Invalid training negative index")
    return np.abs(masses[hard] - masses[:, None]) <= np.maximum(
        masses[:, None] * 35e-6, 0.006
    )
