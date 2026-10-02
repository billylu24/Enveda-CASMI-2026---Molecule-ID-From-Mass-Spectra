"""Annotate real query spectra and mass-compatible structures without labels.

This is an evidence audit, not a calibrated prediction or performance estimate.
It requires only public structure catalogs and query MS/MS, no model checkpoint.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.chemical_priors import (
    candidate_scores, dictionary_sha256, extract_evidence, load_rules,
)
from casmi_ml.ranking import CandidateIndex, center_mass


def audit(queries, catalog, dictionary, output):
    frame = pd.read_parquet(queries)
    candidates = pd.read_parquet(catalog)
    if 'formal_charge' in candidates.columns:
        candidates = candidates.loc[candidates.formal_charge.eq(0)]
    index = CandidateIndex(candidates)
    rules = load_rules(dictionary)
    key_column = 'molecule_id' if 'molecule_id' in frame.columns else 'inchikey14'
    rows = []
    for key, group in frame.groupby(key_column, sort=True):
        evidence = extract_evidence(group, rules)
        center = center_mass(group)
        pool = index.query(center)
        structures = dict(zip(pool.inchikey14, pool.normalized_smiles))
        scores, support = candidate_scores(structures, evidence, rules)
        rows.append({'molecule_id': key, 'center_mass': center,
                     'spectra': len(group), 'mass_candidates': len(structures),
                     'evidence': evidence,
                     'supported_candidates': [{'key': k, 'score': scores[k],
                                                'supported_rules': support[k]}
                                               for k in sorted(scores) if scores[k] > 0]})
    summary = {'molecules': len(rows), 'matched_molecules': sum(bool(r['evidence']) for r in rows),
               'dictionary_sha256': dictionary_sha256(dictionary),
               'status': 'evidence_only_no_performance_claim',
               'candidate_charge_policy': 'neutral structures only; original charged entries remain in source catalog',
               'rows': rows}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--dictionary', default='configs/chemical_priors.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summary = audit(args.queries, args.catalog, args.dictionary, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
