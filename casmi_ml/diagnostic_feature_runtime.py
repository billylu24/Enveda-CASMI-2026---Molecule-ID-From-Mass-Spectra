"""Exact maximum diagnostic channels without materializing every matched edge."""

import math

import numpy as np

from casmi_ml.chemistry import RULES, clean_peaks, high_resolution, parse_adduct

DIM = len(RULES) * 3


def maximum_evidence_features(row, precise=True):
    result = np.zeros(DIM, np.float32)
    mz, intensity = clean_peaks(row)
    adduct = row.get("adduct")
    ion = parse_adduct(adduct)
    if (
        not high_resolution(row.get("instrument_type"))
        or not ion
        or ion.molecules != 1
        or abs(ion.charge) != 1
    ):
        return result
    try:
        precursor = float(row.get("precursor_mz", math.nan))
    except (TypeError, ValueError):
        precursor = math.nan
    absolute_da, ppm = 0.002 if precise else 0.5, 10.0
    for i, rule in enumerate(RULES):
        if row.get("ionization_mode") not in rule.modes or adduct not in rule.adducts:
            continue
        if rule.kind == "ion":
            found = np.abs(mz - rule.mass) <= max(absolute_da, rule.mass * ppm * 1e-6)
            if found.any():
                result[i * 3] = np.sqrt(intensity[found].max())
            continue
        if math.isfinite(precursor):
            differences = precursor - mz - rule.mass
            tol = max(absolute_da, abs(precursor) * ppm * 1e-6) + np.maximum(
                absolute_da, mz * ppm * 1e-6
            )
            found = (precursor > mz) & (np.abs(differences) <= tol)
            if found.any():
                result[i * 3 + 1] = np.sqrt(intensity[found].max())
        target = mz + rule.mass
        tolerance = np.maximum(absolute_da, mz * ppm * 1e-6) + np.maximum(
            absolute_da, target * ppm * 1e-6
        )
        lo = np.maximum(np.searchsorted(mz, target - tolerance), np.arange(len(mz)) + 1)
        hi = np.searchsorted(mz, target + tolerance)
        maximum = 0.0
        for index in np.flatnonzero(hi > lo):
            maximum = max(
                maximum, min(intensity[index], intensity[lo[index] : hi[index]].max())
            )
        if maximum:
            result[i * 3 + 2] = np.sqrt(maximum)
    return result
