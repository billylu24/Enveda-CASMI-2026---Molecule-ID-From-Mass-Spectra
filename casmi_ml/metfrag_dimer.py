"""Experimental monomer-product scoring of observed singly charged dimer dissociation."""

import numpy as np

from casmi_ml.chemistry import clean_peaks, high_resolution, neutral_mass, parse_adduct
from casmi_ml.metfrag_monomer import ALIASES, MonomerMetFrag


def monomer_product_row(row):
    ion = parse_adduct(row.get("adduct"))
    if ion is None or ion.molecules != 2 or abs(ion.charge) != 1:
        return None
    monomer_adduct = row["adduct"].replace("[2M", "[M", 1)
    monomer = parse_adduct(monomer_adduct)
    if monomer_adduct not in ALIASES or not high_resolution(row.get("instrument_type")):
        return None
    mass = neutral_mass(row)
    if mass is None:
        return None
    virtual_precursor = mass + monomer.shift
    mz, intensity = clean_peaks(row)
    # A measured charged monomer must support the dissociation hypothesis.
    marker = (
        np.abs(mz - virtual_precursor) <= max(0.002, virtual_precursor * 1e-5)
    ) & (intensity >= 0.05)
    products = mz < virtual_precursor - 0.01
    if not marker.any() or not products.any():
        return None
    return {
        "adduct": monomer_adduct,
        "precursor_mz": virtual_precursor,
        "instrument_type": row.get("instrument_type"),
        "ms2_mzs": mz[products].tolist(),
        "ms2_normalized_intensities": intensity[products].tolist(),
    }


class DimerMetFrag(MonomerMetFrag):
    def score(self, row, candidates):
        ion = parse_adduct(row.get("adduct"))
        if ion and ion.molecules == 2:
            transformed = monomer_product_row(row)
            if transformed is None:
                return {"status": "unsupported_dimer_dissociation", "scores": {}}
            return super().score(transformed, candidates)
        return super().score(row, candidates)
