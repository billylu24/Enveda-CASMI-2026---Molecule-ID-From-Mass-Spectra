"""Additional broad MS/MS hypotheses for training-only specificity audits.

Exact ion compositions identify an observation, not a unique fragment or motif.
References describe general fragmentation classes; the SMARTS here are broad
support hypotheses that require empirical training and full-system controls.
"""

import math
from functools import lru_cache

import numpy as np
from rdkit import Chem

from casmi_ml.chemistry import (
    ELECTRON,
    GENERAL,
    PROTONATED,
    Rule,
    clean_peaks,
    composition_mass,
    high_resolution,
    parse_adduct,
)

POSITIVE = ("[M+H]+",)
NEGATIVE = ("[M-H]-",)
PHOSPHATE = "[P](=[O])([O])([O])[O]"


def ion_rule(name, formula, charge, smarts, interpretation):
    return Rule(
        name,
        "ion",
        composition_mass(formula) - charge * ELECTRON,
        ("positive",) if charge > 0 else ("negative",),
        POSITIVE if charge > 0 else NEGATIVE,
        smarts,
        1.0,
        GENERAL,
        interpretation + "; broad hypothesis, rearrangements and isobars possible.",
    )


def loss_rule(name, formula, smarts, interpretation):
    return Rule(
        name,
        "loss",
        composition_mass(formula),
        ("positive", "negative"),
        PROTONATED,
        smarts,
        1.0,
        GENERAL,
        interpretation + "; broad support only, not structural proof.",
    )


RULES = (
    ion_rule(
        "phenyl_ion", "C6H5", 1, "c1ccccc1", "Aromatic-ring-related positive fragment"
    ),
    ion_rule(
        "benzyl_tropylium_ion",
        "C7H7",
        1,
        "c1ccccc1",
        "Aromatic positive fragment, ring rearrangement allowed",
    ),
    ion_rule(
        "benzoyl_ion",
        "C7H5O",
        1,
        "c1ccccc1[C](=[O])",
        "Aromatic-carbonyl-related positive fragment",
    ),
    ion_rule(
        "benzoate_ion",
        "C7H5O2",
        -1,
        "c1ccccc1[C](=[O])[O]",
        "Aromatic-carboxyl-related negative fragment",
    ),
    ion_rule(
        "phosphate_ion", "PO3", -1, PHOSPHATE, "Phosphate-related negative fragment"
    ),
    ion_rule(
        "dihydrogen_phosphate_ion",
        "H2PO4",
        -1,
        PHOSPHATE,
        "Phosphate-related negative fragment",
    ),
    loss_rule("methanol", "CH4O", "[O][CH3]", "Methoxy-related neutral loss"),
    loss_rule(
        "sulfur_dioxide",
        "SO2",
        "[S](=[O])(=[O])",
        "Sulfone or sulfonyl-related neutral loss",
    ),
    loss_rule(
        "sulfur_trioxide",
        "SO3",
        "[S](=[O])(=[O])[O]",
        "Sulfate or sulfonate-related neutral loss",
    ),
    loss_rule("phosphoric_acid", "H3PO4", PHOSPHATE, "Phosphate-related neutral loss"),
    loss_rule(
        "metaphosphoric_acid", "HPO3", PHOSPHATE, "Phosphate-related neutral loss"
    ),
    loss_rule(
        "acetic_acid", "C2H4O2", "[CH3][C](=[O])[O]", "Acetate-related neutral loss"
    ),
)
PATTERNS = tuple(Chem.MolFromSmarts(rule.smarts) for rule in RULES)
DIM = len(RULES) * 3


@lru_cache(maxsize=100000)
def structural_support(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid structure for extended diagnostic support")
    return tuple(bool(molecule.HasSubstructMatch(p)) for p in PATTERNS)


def evidence_features(row):
    """Reuse frozen tolerances and maximal-edge pooling for these new rules only."""
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
    absolute_da, ppm = 0.002, 10.0
    for i, rule in enumerate(RULES):
        if row.get("ionization_mode") not in rule.modes or adduct not in rule.adducts:
            continue
        if rule.kind == "ion":
            found = np.abs(mz - rule.mass) <= max(absolute_da, rule.mass * ppm * 1e-6)
            if found.any():
                result[i * 3] = np.sqrt(intensity[found].max())
            continue
        if math.isfinite(precursor):
            difference = precursor - mz - rule.mass
            tolerance = max(absolute_da, abs(precursor) * ppm * 1e-6) + np.maximum(
                absolute_da, mz * ppm * 1e-6
            )
            found = (precursor > mz) & (np.abs(difference) <= tolerance)
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
