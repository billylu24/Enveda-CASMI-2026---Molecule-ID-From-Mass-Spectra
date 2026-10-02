"""Label-independent generation conditions with a fixed total trajectory budget."""

import json

import numpy as np

from casmi_ml.chemistry import clean_peaks
from casmi_ml.ranking import spectrum_signature


def informative_spectrum(group):
    """Select the largest intensity-effective peak count, with content-only ties."""
    choices = []
    for position, row in enumerate(group.to_dict("records")):
        mz, intensity = clean_peaks(row)
        total = intensity.sum()
        probabilities = intensity / total if total > 0 else np.zeros(len(intensity))
        positive = probabilities[probabilities > 0]
        effective = (
            float(np.exp(-(positive * np.log(positive)).sum()))
            if len(positive)
            else 0.0
        )
        signature = json.dumps(
            {
                "adduct": str(row.get("adduct")),
                "ionization_mode": str(row.get("ionization_mode")),
                "instrument_type": str(row.get("instrument_type")),
                "collision_energy_ev": str(row.get("collision_energy_ev")),
                "precursor_mz": float(row.get("precursor_mz", 0)),
                "spectrum": spectrum_signature(mz, intensity).hex(),
            },
            sort_keys=True,
        )
        choices.append((-effective, signature, position))
    if not choices:
        raise ValueError("Generation needs at least one spectrum")
    return group.iloc[[min(choices)[2]]]


def generation_views(group, samples=128):
    if samples < 2:
        raise ValueError("At least two trajectories required for a view mixture")
    half = samples // 2
    return [(group, half), (informative_spectrum(group), samples - half)]
