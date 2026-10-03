"""Second measured spectrum per training molecule, same observed adduct and mode."""

import numpy as np


def paired_rows(frame, representatives, seed=260):
    representatives = np.asarray(representatives, dtype=np.int64)
    if (
        len(frame) == 0
        or (representatives < 0).any()
        or (representatives >= len(frame)).any()
    ):
        raise ValueError("Representative row indices differ from training data")
    groups = frame.groupby(
        ["inchikey14", "adduct", "ionization_mode"], dropna=False, sort=False
    ).indices
    index_groups = {
        int(index): indexes for indexes in groups.values() for index in indexes
    }
    rng = np.random.default_rng(seed)
    second = []
    for index in representatives:
        candidates = index_groups[int(index)]
        candidates = candidates[candidates != index]
        second.append(int(rng.choice(candidates)) if len(candidates) else int(index))
    return np.asarray(second, dtype=np.int64)
