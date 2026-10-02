"""Shared charge-aware candidate windows; historical retrieval stays unchanged."""

import numpy as np
import pandas as pd

from casmi_ml.chemistry import neutral_mass
from casmi_ml.ranking import center_mass


def mass_centers(group, variant):
    legacy = center_mass(group)
    if variant == "legacy":
        return [] if legacy is None else [legacy]
    if variant not in ["charge_aware_median", "charge_aware_union"]:
        raise ValueError("Unsupported mass hypothesis")
    values = [m for r in group.to_dict("records") if (m := neutral_mass(r)) is not None]
    if not values:
        return [] if legacy is None else [legacy]
    return (
        [float(np.median(values))]
        if variant == "charge_aware_median"
        else sorted(set(values))
    )


def candidate_window(index, group, variant):
    centers = mass_centers(group, variant)
    if not centers:
        return index.fps(index.catalog.iloc[:0])
    frame = pd.concat([index.query(m) for m in centers]).drop_duplicates("inchikey14")
    return index.fps(frame)


def hypothesis_baseline(group, pool, fps, reference, variant):
    from casmi_ml.ranking import baseline_rank, rrf

    centers = mass_centers(group, variant)
    if not centers:
        return pool.inchikey14.tolist()
    rankings = [
        baseline_rank(group, pool, fps, reference, center) for center in centers
    ]
    return rankings[0] if len(rankings) == 1 else rrf(rankings)
