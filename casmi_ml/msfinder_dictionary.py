"""Strict, provenance-aware import and peak annotation for MS-FINDER TSVs.

The upstream ``Exact mass`` column contains atomic formula masses, including
in ProductIonLib_vs1.pid.  FragmentAssigner adds an electron mass to observed
positive-ion m/z (subtracts for negative ions).  Therefore this adapter converts
atomic product-ion mass to singly charged m/z by subtracting / adding an electron.
Neutral-loss masses receive no electron correction.  Callers must explicitly
select the input convention; the formula check rejects an incompatible choice.

Upstream implementation:
https://github.com/systemsomicslab/MsdialWorkbench/blob/master/src/Common/CommonStandard/FormulaGenerator/Parser/FragmentDbParser.cs
https://github.com/systemsomicslab/MsdialWorkbench/blob/master/src/Common/CommonStandard/FormulaGenerator/Function/FragmentAssigner.cs

These dictionaries associate formulas with fragment structure keys.  They do
not provide SMARTS or validate a functional group in a candidate parent molecule.
Annotations below are mass matches only and never enable candidate reranking.
"""

import argparse
from bisect import bisect_left, bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse


ELECTRON_MASS = 0.0005485799  # Matches upstream FragmentAssigner.cs.
HEADERS = ("Exact mass", "Formula", "IonMode", "Frequency", "FragmentShortInChIKeys")
MASS_CONVENTIONS = ("msfinder_atomic", "already_mz", "neutral_mass")
# A consistency check, not a replacement for the source's numerical masses.
ATOMIC_MASSES = {
    "H": 1.00782503223, "C": 12.0, "N": 14.00307400443,
    "O": 15.99491461957, "P": 30.97376199842, "S": 31.9720711744,
    "F": 18.99840316273, "Cl": 34.968852682, "Br": 78.9183376,
    "I": 126.904468, "Si": 27.97692653465,
}
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")
KEY_PATTERN = re.compile(r"[A-Z]{14}\Z")
FORMULA_MASS_TOLERANCE = 0.00002  # Accommodates historical atom-mass constants.


def _mode(value):
    modes = {"positive": "positive", "negative": "negative"}
    try:
        return modes[str(value).strip().lower()]
    except KeyError as error:
        raise ValueError(f"invalid ion mode: {value!r}") from error


def _kind(value):
    kinds = {"diagnostic_ion": "diagnostic_ion", "product_ion": "diagnostic_ion",
             "neutral_loss": "neutral_loss"}
    try:
        return kinds[value]
    except KeyError as error:
        raise ValueError("kind must be diagnostic_ion, product_ion, or neutral_loss") from error


def _formula_mass(formula):
    position, mass, seen = 0, 0.0, set()
    for match in FORMULA_PATTERN.finditer(formula):
        element = match.group(1)
        if match.start() != position or element not in ATOMIC_MASSES or element in seen:
            raise ValueError(f"unsupported or malformed formula: {formula!r}")
        mass += ATOMIC_MASSES[element] * int(match.group(2) or "1")
        position = match.end()
        seen.add(element)
    if not formula or position != len(formula):
        raise ValueError(f"unsupported or malformed formula: {formula!r}")
    return mass


def _finite_number(value, field, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"invalid {field}: {value!r}")
    return number


def load_msfinder_dictionary(path, kind, *, mass_convention=None):
    """Return normalized rules; refuse to guess the input mass convention.

    ``msfinder_atomic`` is the verified convention of the distributed .pid/.ndb
    files. ``already_mz`` is accepted only for product-ion files whose masses
    have already been electron-corrected. ``neutral_mass`` is neutral-loss only.
    Numerical input masses are retained in ``stored_mass`` and any correction
    is recorded explicitly. Frequency is source metadata, not a probability.
    Malformed rows cause a line-numbered error rather than a partial import.
    """
    kind = _kind(kind)
    if mass_convention not in MASS_CONVENTIONS:
        raise ValueError(f"explicit mass_convention required: {MASS_CONVENTIONS}")
    if kind == "diagnostic_ion" and mass_convention == "neutral_mass":
        raise ValueError("neutral_mass is not a product-ion convention")
    if kind == "neutral_loss" and mass_convention == "already_mz":
        raise ValueError("neutral-loss masses are not ion m/z")
    path = Path(path)
    rules = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != list(HEADERS):
            raise ValueError(f"{path}:1: expected TSV columns {HEADERS}")
        for row in reader:
            # Blank lines at the end are harmless; blank cells in a row are not.
            if not row or all(not field.strip() for field in row):
                continue
            try:
                if len(row) != len(HEADERS):
                    raise ValueError("expected exactly five TSV cells")
                stored_mass = _finite_number(row[0], "Exact mass", positive=True)
                formula = row[1].strip()
                atomic_mass = _formula_mass(formula)
                mode = _mode(row[2])
                frequency = _finite_number(row[3], "Frequency")
                fragment_keys = [key.strip() for key in row[4].split(";")]
                if not all(KEY_PATTERN.fullmatch(key) for key in fragment_keys):
                    raise ValueError("invalid FragmentShortInChIKeys (expected 14 uppercase letters)")
                fragment_keys = list(dict.fromkeys(fragment_keys))
                correction = 0.0
                if kind == "diagnostic_ion":
                    correction = -ELECTRON_MASS if mode == "positive" else ELECTRON_MASS
                expected = atomic_mass + (correction if mass_convention == "already_mz" else 0)
                if abs(stored_mass - expected) > FORMULA_MASS_TOLERANCE:
                    raise ValueError(
                        f"Exact mass disagrees with {formula} under {mass_convention}: "
                        f"{stored_mass:.9f} vs {expected:.9f}"
                    )
                applied_correction = correction if mass_convention == "msfinder_atomic" else 0.0
                rules.append({
                    "id": f"msfinder:{kind}:{mode}:{formula}:{reader.line_num}",
                    "kind": kind, "mode": mode,
                    "mass": stored_mass + applied_correction,
                    "stored_mass": stored_mass, "formula": formula,
                    "frequency": frequency, "fragment_keys": fragment_keys,
                    "mass_convention": mass_convention,
                    "mass_correction_da": applied_correction,
                    "source_line": reader.line_num,
                })
            except ValueError as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error
    if not rules:
        raise ValueError(f"{path}: no dictionary records")
    return rules


def match_msfinder_dictionary(mzs, intensities, rules, *, mode, precursor_mz=None,
                             precursor_charge=1, ppm=10.0, da_floor=0.002,
                             min_relative_intensity=0.01):
    """Annotate product ions and precursor-to-peak apparent neutral losses.

    Product ions are assumed singly charged. Losses require a singly charged
    precursor and fragment; other precursor charges are rejected when loss
    rules are active. Ion tolerance is max(da_floor, ppm * reference m/z / 1e6).
    Loss tolerance conservatively sums the precursor and fragment ppm errors,
    with da_floor as the minimum tolerance on the difference. Every matching
    formula is retained, including ambiguous alternatives. Frequency and keys
    are returned as metadata; no functional-group or parent identity is inferred.
    """
    mode = _mode(mode)
    ppm = _finite_number(ppm, "ppm")
    da_floor = _finite_number(da_floor, "da_floor")
    threshold = _finite_number(min_relative_intensity, "min_relative_intensity")
    if threshold > 1 or (ppm == 0 and da_floor == 0):
        raise ValueError("relative intensity must be <=1 and a mass tolerance must be positive")
    mzs, intensities, rules = list(mzs), list(intensities), list(rules)
    if len(mzs) != len(intensities):
        raise ValueError("mzs and intensities must have the same length")
    active = [rule for rule in rules if rule["mode"] == mode]
    needs_losses = any(rule["kind"] == "neutral_loss" for rule in active)
    if needs_losses and precursor_mz is not None and precursor_charge != 1:
        raise ValueError("precursor-to-peak neutral losses require precursor_charge=1")
    if precursor_mz is not None:
        precursor_mz = _finite_number(precursor_mz, "precursor_mz", positive=True)
    peaks = []
    for index, (mz, intensity) in enumerate(zip(mzs, intensities)):
        mz = _finite_number(mz, f"peak {index} m/z", positive=True)
        intensity = _finite_number(intensity, f"peak {index} intensity")
        peaks.append((mz, intensity, index))
    maximum = max((peak[1] for peak in peaks), default=0)
    if maximum == 0:
        return []
    peaks = sorted(peak for peak in peaks if peak[1] > 0 and peak[1] >= threshold * maximum)
    masses = [peak[0] for peak in peaks]
    matches = []
    for rule in active:
        reference = rule["mass"]
        if rule["kind"] == "diagnostic_ion":
            target = reference
            tolerance = max(da_floor, reference * ppm * 1e-6)
        elif rule["kind"] == "neutral_loss":
            if precursor_mz is None:
                continue
            target = precursor_mz - reference
            if target <= 0:
                continue
            tolerance = max(da_floor, (precursor_mz + target) * ppm * 1e-6)
        else:
            raise ValueError(f"unsupported rule kind: {rule['kind']!r}")
        start, end = bisect_left(masses, target - tolerance), bisect_right(masses, target + tolerance)
        for mz, intensity, index in peaks[start:end]:
            observed = mz if rule["kind"] == "diagnostic_ion" else precursor_mz - mz
            matches.append({
                "rule_id": rule["id"], "kind": rule["kind"], "mode": mode,
                "formula": rule["formula"], "frequency": rule["frequency"],
                "fragment_keys": list(rule["fragment_keys"]),
                "peak_index": index, "peak_mz": mz,
                "relative_intensity": intensity / maximum,
                "reference_mass": reference, "observed_mass": observed,
                "error_da": observed - reference,
                "error_ppm": (observed - reference) / reference * 1e6,
                "tolerance_da": tolerance,
                "evidence_type": "mass_match_only",
            })
    return sorted(matches, key=lambda match: (match["peak_index"], match["kind"], match["rule_id"]))


def dictionary_manifest(path, rules, *, source_uri, license_status, license_uri=None):
    """Record independently supplied database provenance and license status."""
    if not source_uri or not urlparse(source_uri).scheme:
        raise ValueError("source_uri must be a nonempty URI")
    if license_status not in ("unverified", "verified", "restricted"):
        raise ValueError("license_status must be unverified, verified, or restricted")
    if license_status == "verified" and (not license_uri or not urlparse(license_uri).scheme):
        raise ValueError("verified license status requires a license_uri")
    return {
        "format": "msfinder_dictionary_v1", "source_uri": source_uri,
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "license_status": license_status, "license_uri": license_uri,
        "records": len(rules),
        "mass_conventions": sorted({rule["mass_convention"] for rule in rules}),
        "electron_mass_da": ELECTRON_MASS,
        "mode_counts": {mode: sum(rule["mode"] == mode for rule in rules)
                        for mode in ("positive", "negative")},
        "annotation_scope": "mass matches only; no candidate SMARTS mapping or ranking",
        "license_note": "Software license is not assumed to license the resource data.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=("diagnostic_ion", "product_ion", "neutral_loss"), required=True)
    parser.add_argument("--mass-convention", choices=MASS_CONVENTIONS, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--license-status", choices=("unverified", "verified", "restricted"), required=True)
    parser.add_argument("--license-uri")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spectra", type=Path, help="Optional JSON list of spectra (mzs, intensities, mode, precursor_mz)")
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--da-floor", type=float, default=0.002)
    args = parser.parse_args(argv)
    rules = load_msfinder_dictionary(args.input, args.kind, mass_convention=args.mass_convention)
    manifest = dictionary_manifest(args.input, rules, source_uri=args.source_uri,
                                   license_status=args.license_status, license_uri=args.license_uri)
    result = {"manifest": manifest, "rules": rules}
    if args.spectra:
        spectra = json.loads(args.spectra.read_text())
        result["annotations"] = [{
            "spectrum_id": spectrum.get("spectrum_id", index),
            "matches": match_msfinder_dictionary(
                spectrum["mzs"], spectrum["intensities"], rules,
                mode=spectrum["mode"], precursor_mz=spectrum.get("precursor_mz"),
                precursor_charge=spectrum.get("precursor_charge", 1),
                ppm=args.ppm, da_floor=args.da_floor),
        } for index, spectrum in enumerate(spectra)]
        manifest["spectra_sha256"] = hashlib.sha256(args.spectra.read_bytes()).hexdigest()
        manifest["matching"] = {"ppm": args.ppm, "da_floor": args.da_floor,
                                "min_relative_intensity": 0.01,
                                "neutral_loss_tolerance": "sum of precursor and fragment ppm errors"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
