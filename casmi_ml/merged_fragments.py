"""Observable peak union across compatible high-resolution monomer spectra."""

import numpy as np

from casmi_ml.chemistry import clean_peaks, high_resolution, neutral_mass
from casmi_ml.metfrag import score_group
from casmi_ml.metfrag_monomer import ALIASES


def merge_records(records):
    groups, untouched = [], []
    for row in records:
        row = {
            k: row[k]
            for k in (
                "adduct",
                "precursor_mz",
                "instrument_type",
                "ionization_mode",
                "ms2_mzs",
                "ms2_normalized_intensities",
            )
            if k in row
        }
        mass = neutral_mass(row)
        alias = ALIASES.get(row.get("adduct"))
        if (
            alias is None
            or mass is None
            or not high_resolution(row.get("instrument_type"))
        ):
            untouched.append(row)
            continue
        group = next(
            (
                g
                for g in groups
                if g[0] == alias and abs(g[1] - mass) <= max(0.002, mass * 10e-6)
            ),
            None,
        )
        if group is None:
            group = [alias, mass, []]
            groups.append(group)
        group[2].append(row)
    result = list(untouched)
    for alias, _, rows in groups:
        if len(rows) == 1:
            result.append(rows[0])
            continue
        mz, intensity = [], []
        for row in rows:
            masses, values = clean_peaks(row)
            valid = masses < float(row["precursor_mz"]) - 0.01
            mz.extend(masses[valid].tolist())
            intensity.extend(values[valid].tolist())
        order = np.argsort(mz, kind="stable")
        merged = []
        for i in order:
            mass, value = mz[i], intensity[i]
            if merged and abs(mass - merged[-1][0]) <= max(0.002, mass * 10e-6):
                # Keep one exact observed mass, never drift a chain of clustered peaks.
                merged[-1][1] = max(merged[-1][1], value)
            else:
                merged.append([mass, value])
        row = rows[0]
        result.append(
            {
                "adduct": row["adduct"],
                "precursor_mz": float(
                    np.median([float(r["precursor_mz"]) for r in rows])
                ),
                "instrument_type": row["instrument_type"],
                "ionization_mode": row.get("ionization_mode"),
                "ms2_mzs": [v[0] for v in merged],
                "ms2_normalized_intensities": [v[1] for v in merged],
            }
        )
    return result


def score_group_merged(fragmenter, records, candidates, deadline=None):
    """Separate ions stay separate; fewer repeated fragment enumerations per ion."""
    merged = merge_records(records)
    fragmenter.merged_input_spectra = getattr(
        fragmenter, "merged_input_spectra", 0
    ) + len(records)
    fragmenter.merged_engine_spectra = getattr(
        fragmenter, "merged_engine_spectra", 0
    ) + len(merged)
    return score_group(fragmenter, merged, candidates, deadline)
