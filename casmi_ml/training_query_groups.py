"""All original measured training rows per molecule, with explicit padding masks."""

import numpy as np


def grouped_rows(frame, representative):
    groups = frame.groupby("inchikey14", sort=True).indices
    representative = np.asarray(representative, dtype=np.int64)
    if len(representative) != len(groups):
        raise ValueError(
            "One original representative per ordered training molecule required"
        )
    width = max(len(ids) for ids in groups.values())
    indexes = np.empty((len(groups), width), dtype=np.int64)
    valid = np.zeros_like(indexes, dtype=bool)
    for i, (ids, selected) in enumerate(zip(groups.values(), representative)):
        if selected not in ids:
            raise ValueError("Representative belongs to another molecule")
        indexes[i] = selected
        indexes[i, : len(ids)] = ids
        valid[i, : len(ids)] = True
    return indexes, valid


def masked_group_mean(values, valid):
    if values.shape[:-1] != valid.shape or not bool(valid.any(-1).all()):
        raise ValueError("Every aligned spectrum group requires a valid row")
    return (values * valid.unsqueeze(-1)).sum(-2) / valid.sum(-1, keepdim=True)
