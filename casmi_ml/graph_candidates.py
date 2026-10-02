"""Bounded graph-edit extrapolation for a shared neural candidate pool.

This module is a deterministic chemical graph edit operator, not a learned GAN
graph decoder. It swaps equal-order non-aromatic bonds, preserving atoms,
charges and each atom's degree/bond-order sum. Sanitized outputs must preserve
the anchor's exact formula and mass. Novelty means absent from the caller's
supplied identity set; it does not establish absence from PubChem or nature.
"""

from functools import lru_cache
import hashlib
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


MAX_ANCHORS = 2
ATTEMPTS_PER_ANCHOR = 256
MAX_NEW = 32
ENUMERATOR = rdMolStandardize.TautomerEnumerator()


def _identity(mol):
    canonical = ENUMERATOR.Canonicalize(mol)
    key = Chem.MolToInchiKey(canonical)[:14]
    if len(key) != 14:
        raise ValueError("Cannot determine an official graph identity")
    return key, Chem.MolToSmiles(canonical), canonical


def _atom_invariants(mol):
    return tuple((atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetDegree(),
                  sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())) for atom in mol.GetAtoms())


def group_seed(group, seed=20261002):
    """Hash query content only; exclude labels, keys, and query ordering."""
    records = group.to_dict("records") if hasattr(group, "to_dict") else list(group)
    digests = []
    for row in records:
        digest = hashlib.sha256()
        for name in ["adduct", "ionization_mode"]:
            digest.update(str(row.get(name)).encode() + b"\0")
        precursor = row.get("precursor_mz")
        digest.update(np.asarray([np.nan if precursor is None else precursor], dtype=np.float64).tobytes())
        for name in ["ms2_mzs", "ms2_normalized_intensities"]:
            values = np.asarray(row[name], dtype=np.float64)
            digest.update(np.asarray([len(values)], dtype=np.int64).tobytes())
            digest.update(values.tobytes())
        digests.append(digest.digest())
    digest = hashlib.sha256(f"casmi-graph-edits-v1:{seed}:".encode() + b"".join(sorted(digests))).digest()
    return int.from_bytes(digest[:8], "big")


@lru_cache(maxsize=512)
def _anchor_edits(smiles, seed, attempts):
    """Cache raw unique edits before the caller's catalog novelty filtering."""
    mol = Chem.MolFromSmiles(smiles)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mass = float(Descriptors.ExactMolWt(mol))
    invariants = _atom_invariants(mol)
    anchor_id, _, _ = _identity(mol)
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
             if bond.GetBondType() == Chem.BondType.SINGLE and not bond.GetIsAromatic()]
    stats = {"attempted_edits": 0, "overlapping_endpoints": 0, "duplicate_bond": 0,
             "invalid": 0, "disconnected": 0, "formula_mismatch": 0,
             "mass_mismatch": 0, "degree_mismatch": 0, "collapsed": 0}
    if len(bonds) < 2:
        return (), stats
    derived = hashlib.sha256(f"{seed}:{smiles}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(derived[:8], "big"))
    output, seen = [], {anchor_id}
    for _ in range(attempts):
        stats["attempted_edits"] += 1
        first, second = rng.choice(len(bonds), size=2, replace=False)
        a, b = bonds[int(first)]
        c, d = bonds[int(second)]
        if len({a, b, c, d}) != 4:
            stats["overlapping_endpoints"] += 1
            continue
        proposed = [(a, d), (c, b)] if rng.integers(2) else [(a, c), (b, d)]
        if any(mol.GetBondBetweenAtoms(left, right) is not None for left, right in proposed):
            stats["duplicate_bond"] += 1
            continue
        edited = Chem.RWMol(mol)
        edited.RemoveBond(a, b)
        edited.RemoveBond(c, d)
        for left, right in proposed:
            edited.AddBond(left, right, Chem.BondType.SINGLE)
        candidate = edited.GetMol()
        try:
            Chem.SanitizeMol(candidate)
            Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
            if len(Chem.GetMolFrags(candidate)) != 1:
                stats["disconnected"] += 1
                continue
            if _atom_invariants(candidate) != invariants:
                stats["degree_mismatch"] += 1
                continue
            if rdMolDescriptors.CalcMolFormula(candidate) != formula:
                stats["formula_mismatch"] += 1
                continue
            candidate_mass = float(Descriptors.ExactMolWt(candidate))
            if not math.isfinite(candidate_mass) or abs(candidate_mass-mass) > 1e-8:
                stats["mass_mismatch"] += 1
                continue
            identity, normalized, canonical = _identity(candidate)
            if (rdMolDescriptors.CalcMolFormula(canonical) != formula
                    or abs(float(Descriptors.ExactMolWt(canonical))-mass) > 1e-8):
                stats["formula_mismatch"] += 1
                continue
        except (RuntimeError, ValueError):
            stats["invalid"] += 1
            continue
        if identity in seen:
            stats["collapsed"] += 1
            continue
        seen.add(identity)
        output.append((identity, normalized, candidate_mass, Chem.GetFormalCharge(canonical), formula))
    return tuple(output), stats


def generate_candidates(anchors, excluded_identities, seed=20261002,
                        *, attempts=ATTEMPTS_PER_ANCHOR, max_new=MAX_NEW):
    """Return novel bounded graph edits and auditable rejection counts.

    anchors is a ranked list of SMILES; only its first two distinct valid
    official identities are used. excluded_identities should cover the entire
    supplied reference/public catalog, not merely its top-ranked structures.
    An unchanged or tautomer-equivalent anchor is never a new candidate.
    """
    if not isinstance(attempts, int) or not 0 <= attempts <= ATTEMPTS_PER_ANCHOR:
        raise ValueError("Graph attempts must be an integer in 0..256")
    if not isinstance(max_new, int) or not 0 <= max_new <= MAX_NEW:
        raise ValueError("New graph limit must be an integer in 0..32")
    excluded = set(excluded_identities)
    anchor_rows, invalid_anchors, anchor_seen = [], 0, set()
    for smiles in anchors:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
            invalid_anchors += 1
            continue
        try:
            identity, _, _ = _identity(mol)
        except (RuntimeError, ValueError):
            invalid_anchors += 1
            continue
        if identity in anchor_seen:
            continue
        anchor_seen.add(identity)
        # Retain the actual anchor graph for formula/degree-preserving edits.
        anchor_rows.append((identity, Chem.MolToSmiles(mol)))
        if len(anchor_rows) == MAX_ANCHORS:
            break
    output, generated_seen = [], set()
    totals = {"attempted_edits": 0, "overlapping_endpoints": 0, "duplicate_bond": 0,
              "invalid": 0, "disconnected": 0, "formula_mismatch": 0,
              "mass_mismatch": 0, "degree_mismatch": 0, "collapsed": 0,
              "excluded_existing": 0, "offered_unique_edits": 0, "cache_hits": 0}
    if max_new:
        for anchor_identity, smiles in anchor_rows:
            before = _anchor_edits.cache_info().hits
            edits, stats = _anchor_edits(smiles, int(seed), attempts)
            totals["cache_hits"] += _anchor_edits.cache_info().hits - before
            for name, count in stats.items():
                totals[name] += count
            totals["offered_unique_edits"] += len(edits)
            for identity, normalized, mass, charge, formula in edits:
                if identity in excluded or identity in anchor_seen:
                    totals["excluded_existing"] += 1
                    continue
                if identity in generated_seen:
                    totals["collapsed"] += 1
                    continue
                generated_seen.add(identity)
                output.append({"inchikey14": identity, "normalized_smiles": normalized,
                               "mass": mass, "identity": identity, "formal_charge": charge,
                               "molecular_formula": formula, "origin": "graph_edit",
                               "anchor_identity": anchor_identity})
                if len(output) == max_new:
                    break
            if len(output) == max_new:
                break
    report = {**totals, "invalid_anchors": invalid_anchors,
              "anchor_ids": [value for value, _ in anchor_rows],
              "anchors_used": len(anchor_rows), "attempts_per_anchor_limit": attempts,
              "maximum_new_graphs": max_new, "new_graphs": len(output), "novel": len(output),
              "seed": int(seed), "learned_graph_decoder": False,
              "constraint": "single non-aromatic equal-order bond swaps; exact formula/mass, charge, per-atom degree/bond-order preserved",
              "novelty_definition": "Absent from caller-supplied excluded identities and both anchors; external databases not independently searched."}
    return output, report
