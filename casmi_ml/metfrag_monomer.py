"""Experimental exact monomer aliases supported by the pinned MetFrag ion table."""

import fcntl
import hashlib
import json

from casmi_ml.chemistry import clean_peaks, high_resolution, neutral_mass, parse_adduct
from casmi_ml.metfrag import MetFrag

# Verified in Constants.ADDUCT_TYPES of the pinned 2.6.11 jar. Do not map dimers,
# multiply charged ions, dehydrated precursors, or unspecified radical ions.
ALIASES = {
    "[M+H]+": "[M+H]+",
    "[M-H]-": "[M-H]-",
    "[M+Na]+": "[M+Na]+",
    "[M+K]+": "[M+K]+",
    "[M+NH4]+": "[M+NH4]+",
    "[M+Cl]-": "[M+Cl]-",
    "[M+CH2O2-H]-": "[M+HCOO]-",
    "[M+C2H4O2-H]-": "[M+CH3COO]-",
    "[M+CH3OH+H]+": "[M+CH3OH+H]+",
}


class MonomerMetFrag(MetFrag):
    def score(self, row, candidates):
        adduct = ALIASES.get(row.get("adduct"))
        mass = neutral_mass(row)
        if (
            adduct is None
            or mass is None
            or not high_resolution(row.get("instrument_type"))
        ):
            return {"status": "unsupported_ion_or_resolution", "scores": {}}
        raw_ion, engine_ion = parse_adduct(row["adduct"]), parse_adduct(adduct)
        if raw_ion != engine_ion:
            raise ValueError("MetFrag alias changes precursor composition")
        mz, intensity = clean_peaks(row)
        valid = mz < float(row["precursor_mz"]) - 0.01
        peaks = list(zip(mz[valid].tolist(), intensity[valid].tolist()))
        if not peaks or not candidates:
            return {"status": "empty", "scores": {}}
        payload = {
            "jar": self.sha256,
            "peaks": peaks,
            "mass": mass,
            "adduct": adduct,
            "candidates": sorted(candidates.items()),
            "ppm": 10,
            "absolute_da": 0.002,
            "depth": 2,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache / f"{key}.json"
        with (self.cache / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._run(cache_path, candidates, peaks, mass, adduct)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
