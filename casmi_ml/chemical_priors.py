"""Traceable, positive-only chemistry evidence for offline candidate reranking.

Observed fragments are evidence, not unique assignments of functional groups.
Missing peaks never eliminate a candidate. Weights must be frozen on development
data before an independent molecule-disjoint acceptance test.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    mass: float
    formula: str
    modes: tuple
    adducts: tuple
    smarts: tuple
    weight: float
    references: tuple
    caveat: str


def load_rules(path):
    """Validate the entire dictionary before processing any spectra."""
    document = json.loads(Path(path).read_text())
    if document.get('version') != 1:
        raise ValueError('Unsupported chemical rule dictionary version')
    rules, seen = [], set()
    for raw in document['rules']:
        rule = Rule(**{**raw, **{k: tuple(raw[k]) for k in
                      ['modes', 'adducts', 'smarts', 'references']}})
        if not rule.id or rule.id in seen:
            raise ValueError('Rule IDs must be nonempty and unique')
        if rule.kind not in ['diagnostic_ion', 'neutral_loss']:
            raise ValueError(f'Invalid rule kind: {rule.id}')
        if not math.isfinite(rule.mass) or rule.mass <= 0:
            raise ValueError(f'Invalid mass: {rule.id}')
        if not math.isfinite(rule.weight) or not 0 < rule.weight <= 1:
            raise ValueError(f'Invalid evidence weight: {rule.id}')
        if not rule.modes or not set(rule.modes) <= {'positive', 'negative'}:
            raise ValueError(f'Explicit ion modes required: {rule.id}')
        if not rule.adducts or not rule.references or not rule.caveat:
            raise ValueError(f'Adduct applicability, reference and caveat required: {rule.id}')
        if not rule.smarts or any(Chem.MolFromSmarts(s) is None for s in rule.smarts):
            raise ValueError(f'Invalid or missing structural motif: {rule.id}')
        seen.add(rule.id)
        rules.append(rule)
    return tuple(rules)


def dictionary_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_evidence(group, rules, ppm=10., absolute_tolerance=.002,
                     intensity_floor=.01):
    """Match each rule at most once per spectrum; repeated peaks cannot inflate it.

    Only supported singly charged precursor adducts are considered. Neutral loss
    is precursor m/z minus product m/z, *not* neutral molecular mass minus m/z.
    Fragment-fragment differences and adduct-stripped masses are not guessed.
    """
    if (not all(math.isfinite(v) for v in [ppm, absolute_tolerance, intensity_floor])
            or ppm <= 0 or absolute_tolerance <= 0 or not 0 <= intensity_floor <= 1):
        raise ValueError('Invalid matching tolerances')
    matches = {r.id: [] for r in rules}
    eligible = {r.id: 0 for r in rules}
    rows = group.to_dict('records') if hasattr(group, 'to_dict') else list(group)
    for number, row in enumerate(rows):
        mz = np.asarray(row['ms2_mzs'], dtype=np.float64)
        intensity = np.asarray(row['ms2_normalized_intensities'], dtype=np.float64)
        if mz.ndim != 1 or mz.shape != intensity.shape:
            raise ValueError('Peak mass and intensity arrays must be aligned vectors')
        precursor = row.get('precursor_mz')
        if precursor is None or not np.isfinite(precursor) or precursor <= 0:
            continue
        valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (mz <= precursor + 2.) & (intensity > 0)
        mz, intensity = mz[valid], intensity[valid]
        if not len(mz):
            continue
        intensity = intensity / intensity.max()
        keep = intensity >= intensity_floor
        mz, intensity = mz[keep], intensity[keep]
        for rule in rules:
            if row.get('ionization_mode') not in rule.modes or row.get('adduct') not in rule.adducts:
                continue
            eligible[rule.id] += 1
            values = mz if rule.kind == 'diagnostic_ion' else float(precursor) - mz
            if rule.kind == 'diagnostic_ion':
                tolerance = np.full(len(mz), max(rule.mass * ppm * 1e-6, absolute_tolerance))
            else:
                # A difference inherits uncertainty from both measured masses.
                # Sum bounds conservatively; do not apply ppm only to the loss.
                precursor_bound = max(float(precursor) * ppm * 1e-6, absolute_tolerance)
                tolerance = precursor_bound + np.maximum(mz * ppm * 1e-6, absolute_tolerance)
            errors = np.abs(values - rule.mass)
            ids = np.flatnonzero(errors <= tolerance)
            if not len(ids):
                continue
            # Exact mass closeness and intensity affect positive evidence only.
            strengths = np.sqrt(intensity[ids]) * np.exp(-.5 * (errors[ids] / tolerance[ids]) ** 2)
            best = ids[int(np.argmax(strengths))]
            matches[rule.id].append({
                'spectrum': str(row.get('spectrum_id', number)),
                'adduct': row.get('adduct'), 'mode': row.get('ionization_mode'),
                'precursor_mz': float(precursor), 'fragment_mz': float(mz[best]),
                'observed_mass': float(values[best]),
                'mass_error_da': float(values[best] - rule.mass),
                'relative_intensity': float(intensity[best]),
                'strength': float(strengths.max()),
            })
    output = []
    for rule in rules:
        hits = matches[rule.id]
        if hits:
            # Correlated repeat acquisitions do not add unbounded evidence.
            strength = max(h['strength'] for h in hits)
            output.append({'rule_id': rule.id, 'kind': rule.kind, 'formula': rule.formula,
                           'expected_mass': rule.mass, 'strength': strength,
                           'matched_spectra': len(hits), 'eligible_spectra': eligible[rule.id],
                           'observations': hits, 'references': list(rule.references),
                           'caveat': rule.caveat})
    return output


@lru_cache(maxsize=512)
def _patterns(smarts):
    return tuple(Chem.MolFromSmarts(s) for s in smarts)


def candidate_scores(smiles_by_key, evidence, rules):
    """Score motif support in [0, 1]; neither absence nor mismatch is a hard filter."""
    rule_by_id = {r.id: r for r in rules}
    active = [(rule_by_id[e['rule_id']], float(e['strength'])) for e in evidence]
    total = sum(rule.weight for rule, _ in active)
    scores, supported = {}, {}
    for key, smiles in smiles_by_key.items():
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        ids = []
        numerator = 0.
        if mol is not None and mol.GetNumAtoms():
            for rule, strength in active:
                if any(mol.HasSubstructMatch(pattern) for pattern in _patterns(rule.smarts)):
                    ids.append(rule.id)
                    numerator += rule.weight * strength
        scores[key] = numerator / total if total else 0.
        supported[key] = ids
    return scores, supported


def rerank(base_ranking, scores, weight):
    """Bounded chemistry RRF; ties retain their original rank exactly.

    No peaks, equal chemical evidence, or weight zero preserve the input list.
    Only existing candidate keys are ranked, and no key can be discarded.
    """
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Chemical fusion weight must be between zero and one')
    base = list(dict.fromkeys(base_ranking))
    values = [float(scores.get(k, 0.)) for k in base]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Chemical evidence must be finite and bounded')
    if not base or weight == 0 or max(values) == min(values):
        return base
    # Tied motif supports receive the same rank; key order cannot invent evidence.
    level_rank = {value: 1 + sum(v > value for v in values) for value in set(values)}
    effective_weight = weight * max(values)
    fused = {(key): (1-effective_weight) / (60+rank) + effective_weight / (60+level_rank[value])
             for rank, (key, value) in enumerate(zip(base, values), 1)}
    original = {key: i for i, key in enumerate(base)}
    return sorted(base, key=lambda key: (-fused[key], original[key], key))
