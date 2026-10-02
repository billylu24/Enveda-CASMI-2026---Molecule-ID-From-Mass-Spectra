"""Versioned, conservative MS/MS evidence; observations are not structural proof."""

import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem

from baseline import ATOMIC_MASS

VERSION = 1
ELECTRON = 0.000548579909
FORMULA = re.compile(r"([A-Z][a-z]?)(\d*)")
ADDUCT = re.compile(r"^\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])$")


def composition_mass(formula):
    pos, mass = 0, 0.0
    for m in FORMULA.finditer(formula):
        if m.start() != pos or m[1] not in ATOMIC_MASS:
            raise ValueError(f"Unsupported composition {formula}")
        mass += ATOMIC_MASS[m[1]] * int(m[2] or 1)
        pos = m.end()
    if not formula or pos != len(formula):
        raise ValueError(f"Unsupported composition {formula}")
    return mass


@dataclass(frozen=True)
class Ion:
    molecules: int
    charge: int
    shift: float

    def neutral_mass(self, mz):
        return (abs(self.charge) * float(mz) - self.shift) / self.molecules


@lru_cache(maxsize=256)
def parse_adduct(value):
    """n M + atomic additions/removals, including ionic electron correction."""
    m = ADDUCT.fullmatch(str(value))
    if not m:
        return None
    n, charge = int(m[1] or 1), int(m[3] or 1) * (1 if m[4] == "+" else -1)
    if n < 1 or not 1 <= abs(charge) <= 8:
        return None
    shift = -charge * ELECTRON
    try:
        for part in re.finditer(r"([+-])(\d*)([A-Za-z][A-Za-z0-9]*)", m[2]):
            shift += (
                (1 if part[1] == "+" else -1)
                * int(part[2] or 1)
                * composition_mass(part[3])
            )
    except ValueError:
        return None
    return Ion(n, charge, shift)


def neutral_mass(row):
    ion = parse_adduct(row.get("adduct"))
    try:
        mass = ion.neutral_mass(row["precursor_mz"]) if ion else math.nan
    except (KeyError, TypeError, ValueError):
        mass = math.nan
    return mass if math.isfinite(mass) and mass > 0 else None


def high_resolution(instrument):
    name = str(instrument or "").lower().replace("-", "").replace(" ", "")
    return any(
        s in name for s in ["qtof", "orbitrap", "qexactive", "ft", "timstof", "toftime"]
    )


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    mass: float
    modes: tuple
    adducts: tuple
    smarts: str
    weight: float
    source: str
    interpretation: str


# References describe fragmentation chemistry; SMARTS encode broad support only.
GENERAL = "https://doi.org/10.1007/978-3-642-10711-5"
GLYCOSIDES = "https://doi.org/10.1002/jms.585"
LIPIDS = "https://doi.org/10.1002/mas.20284"
PROTONATED = ("[M+H]+", "[M-H]-", "[M-H2O+H]+", "[M-2H2O+H]+", "[M-H2O-H]-")
RULES = (
    Rule(
        "water",
        "loss",
        composition_mass("H2O"),
        ("positive", "negative"),
        PROTONATED,
        "[O;H1,H2]",
        0.25,
        GENERAL,
        "Non-specific water-loss support; rearrangements are possible.",
    ),
    Rule(
        "ammonia",
        "loss",
        composition_mass("NH3"),
        ("positive",),
        ("[M+H]+",),
        "[N;H1,H2,H3]",
        0.4,
        GENERAL,
        "Nitrogen-containing functionality; not a unique amine assignment.",
    ),
    Rule(
        "carbon_monoxide",
        "loss",
        composition_mass("CO"),
        ("positive", "negative"),
        PROTONATED,
        "[C,c]=[O]",
        0.3,
        GENERAL,
        "Carbonyl-related support, not proof of a carbonyl.",
    ),
    Rule(
        "carbon_dioxide",
        "loss",
        composition_mass("CO2"),
        ("positive", "negative"),
        PROTONATED,
        "[C](=[O])[O]",
        0.5,
        GENERAL,
        "Carboxyl/carbonate-related support.",
    ),
    Rule(
        "hexose_residue",
        "loss",
        composition_mass("C6H10O5"),
        ("positive", "negative"),
        PROTONATED,
        "[O;R]1[C;R][C;R][C;R][C;R][C;R]1",
        0.7,
        GLYCOSIDES,
        "Hexose-residue loss; linkage is not inferred.",
    ),
    Rule(
        "deoxyhexose_residue",
        "loss",
        composition_mass("C6H10O4"),
        ("positive", "negative"),
        PROTONATED,
        "[O;R]1[C;R][C;R][C;R][C;R][C;R]1",
        0.6,
        GLYCOSIDES,
        "Deoxyhexose-residue loss hypothesis.",
    ),
    Rule(
        "pentose_residue",
        "loss",
        composition_mass("C5H8O4"),
        ("positive", "negative"),
        PROTONATED,
        "[O;R]1[C;R][C;R][C;R][C;R]1",
        0.6,
        GLYCOSIDES,
        "Pentose-residue loss hypothesis.",
    ),
    Rule(
        "phosphocholine",
        "ion",
        composition_mass("C5H15NO4P") - ELECTRON,
        ("positive",),
        ("[M+H]+",),
        "[P](=[O])([O])([O])[O]CC[N+](C)(C)C",
        1.0,
        LIPIDS,
        "Phosphocholine-class support, not a unique lipid identity.",
    ),
)


def rule_manifest():
    return {
        "version": VERSION,
        "rules": [asdict(r) for r in RULES],
        "ppm": 10.0,
        "absolute_da": 0.002,
        "missing_peak_penalty": False,
    }


def clean_peaks(row):
    mz = np.asarray(row.get("ms2_mzs", []), dtype=np.float64)
    intensity = np.asarray(row.get("ms2_normalized_intensities", []), dtype=np.float64)
    if mz.shape != intensity.shape or mz.ndim != 1:
        raise ValueError("Peak arrays must have equal one-dimensional shapes")
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    if len(mz):
        intensity /= intensity.max()
    order = np.argsort(mz, kind="stable")
    return mz[order], intensity[order]


def extract_evidence(row, ppm=10.0, absolute_da=0.002):
    if ppm <= 0 or absolute_da <= 0:
        raise ValueError("Positive mass tolerances required")
    mz, intensity = clean_peaks(row)
    ion = parse_adduct(row.get("adduct"))
    precise = high_resolution(row.get("instrument_type"))
    try:
        precursor = float(row.get("precursor_mz", math.nan))
    except (TypeError, ValueError):
        precursor = math.nan
    mode = row.get("ionization_mode")
    observed = []
    # Every raw difference remains an observation. No inferred charge correction.
    raw = [
        {
            "mz": float(m),
            "intensity": float(i),
            "delta_mz": float(precursor - m) if math.isfinite(precursor) else None,
        }
        for m, i in zip(mz, intensity)
    ]
    for rule in RULES:
        if (
            not precise
            or mode not in rule.modes
            or row.get("adduct") not in rule.adducts
        ):
            continue
        if not ion or ion.molecules != 1 or abs(ion.charge) != 1:
            continue
        if rule.kind == "ion":
            differences = mz - rule.mass
            tolerances = np.maximum(absolute_da, rule.mass * ppm * 1e-6)
            ids = np.flatnonzero(np.abs(differences) <= tolerances)
            for idx in ids:
                observed.append(
                    {
                        "rule": rule.name,
                        "kind": "ion",
                        "mz": float(mz[idx]),
                        "error_da": float(differences[idx]),
                        "strength": float(np.sqrt(intensity[idx])),
                        "edge": ["diagnostic", int(idx)],
                    }
                )
        else:
            if math.isfinite(precursor):
                differences = precursor - mz - rule.mass
                # Errors from both measured endpoints accumulate.
                tol = max(absolute_da, abs(precursor) * ppm * 1e-6) + np.maximum(
                    absolute_da, mz * ppm * 1e-6
                )
                for idx in np.flatnonzero(
                    (precursor > mz) & (np.abs(differences) <= tol)
                ):
                    observed.append(
                        {
                            "rule": rule.name,
                            "kind": "precursor_loss",
                            "mz": float(mz[idx]),
                            "error_da": float(differences[idx]),
                            "strength": float(np.sqrt(intensity[idx])),
                            "edge": ["precursor", int(idx)],
                        }
                    )
            # Search rule-specific differences without an O(N^2) dense matrix.
            for idx, mass in enumerate(mz):
                target = mass + rule.mass
                tol = max(absolute_da, mass * ppm * 1e-6) + max(
                    absolute_da, target * ppm * 1e-6
                )
                lo, hi = np.searchsorted(mz, [target - tol, target + tol])
                for j in range(int(lo), int(hi)):
                    if j <= idx:
                        continue
                    observed.append(
                        {
                            "rule": rule.name,
                            "kind": "peak_loss",
                            "mz": float(mass),
                            "error_da": float(mz[j] - mass - rule.mass),
                            "strength": float(
                                np.sqrt(min(intensity[idx], intensity[j]))
                            ),
                            "edge": [int(j), int(idx)],
                        }
                    )
    loss_edges = [x for x in observed if x["kind"] != "ion"]
    combinations = []
    for first in loss_edges:
        for second in loss_edges:
            if first["edge"][1] == second["edge"][0]:
                combinations.append(
                    {
                        "rules": [first["rule"], second["rule"]],
                        "strength": min(first["strength"], second["strength"]),
                    }
                )
    return {
        "version": VERSION,
        "neutral_mass": neutral_mass(row),
        "precise": precise,
        "ion": asdict(ion) if ion else None,
        "observations": raw,
        "matches": observed,
        "combinations": combinations,
        "uninterpreted_reason": None if precise else "unknown_or_low_mass_resolution",
    }


@lru_cache(maxsize=200000)
def candidate_support(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        rule.name: bool(mol.HasSubstructMatch(Chem.MolFromSmarts(rule.smarts)))
        for rule in RULES
    }


def score_candidate(evidences, smiles, component="combined"):
    if component not in ["diagnostic", "loss", "combined"]:
        raise ValueError(component)
    support = candidate_support(smiles)
    if support is None:
        return {
            "valid": False,
            "score": 0.0,
            "diagnostic": 0.0,
            "loss": 0.0,
            "combination": 0.0,
            "supported_rules": [],
        }
    strongest = {}
    combos = {}
    for evidence in evidences:
        for match in evidence["matches"]:
            name = match["rule"]
            strongest[name] = max(strongest.get(name, 0.0), match["strength"])
        for combo in evidence["combinations"]:
            names = tuple(combo["rules"])
            combos[names] = max(combos.get(names, 0.0), combo["strength"])
    # Max across spectra avoids giving duplicated acquisitions extra votes.
    values = {
        r.name: r.weight * strongest.get(r.name, 0.0) for r in RULES if support[r.name]
    }
    diagnostic = sum(values.get(r.name, 0.0) for r in RULES if r.kind == "ion")
    loss = sum(values.get(r.name, 0.0) for r in RULES if r.kind == "loss")
    combination = 0.1 * sum(
        v for names, v in combos.items() if all(support[n] for n in names)
    )
    total = (
        diagnostic
        if component == "diagnostic"
        else loss + combination
        if component == "loss"
        else diagnostic + loss + combination
    )
    return {
        "valid": True,
        "score": total,
        "diagnostic": diagnostic,
        "loss": loss,
        "combination": combination,
        "supported_rules": sorted(n for n, v in values.items() if v > 0),
    }


def rerank(
    base,
    lookup,
    evidences,
    weight=0.25,
    component="combined",
    top_n=100,
    fragment_scores=None,
):
    """Rerank a shortlist; tied/unsupported evidence contributes no rank preference."""
    if not 0 <= weight <= 1 or top_n < 1:
        raise ValueError("Invalid reranking parameters")
    base = list(dict.fromkeys(base))
    if weight == 0:
        return base
    scores = (
        {k: fragment_scores.get(k, 0.0) for k in base[:top_n]}
        if fragment_scores is not None
        else {
            k: score_candidate(evidences, lookup[k], component)["score"]
            for k in base[:top_n]
        }
    )
    if len(set(scores.values())) < 2:
        return base
    # Competition ranking includes tied ranks, so zero evidence never breaks ties.
    rank = {v: 1 + sum(s > v for s in scores.values()) for v in set(scores.values())}
    fused = {
        k: (1 - weight) / (60 + i)
        + (weight / (60 + rank[scores[k]]) if k in scores else 0.0)
        for i, k in enumerate(base, 1)
    }
    return sorted(base, key=lambda k: (-fused[k], base.index(k), k))
