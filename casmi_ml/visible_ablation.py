"""Reproduce the historical visible predictions using the original reference loader."""
import json
from functools import cache

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import sparse

from baseline import ADDUCT_MASS, formula_mass, load_candidates, make_matrix
from casmi_ml.ablation import COCONUT, ROOT, analog_rank
from casmi_ml.data import write_json
from casmi_ml.ranking import (
    CandidateIndex,
    ReferenceIndex,
    baseline_rank,
    build_candidates,
)
from hybrid import blend, coconut_rank, library_rank


@cache
def structure_key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]


def main():
    test = pd.read_parquet('data/test.parquet')
    neutral = test.precursor_mz.to_numpy() - test.adduct.map(ADDUCT_MASS).to_numpy()
    records, _ = load_candidates('data/train.parquet', neutral[np.isfinite(neutral)])
    masses = np.asarray([r[0] for r in records])
    order = np.argsort(masses)
    matrix = make_matrix([r[3] for r in records])
    directory = ROOT / 'visible_reference'
    directory.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({'mass': masses, 'inchikey14': [r[1] for r in records],
                  'normalized_smiles': [r[2] for r in records]}).to_parquet(directory / 'rows.parquet', index=False)
    sparse.save_npz(directory / 'spectra.npz', matrix)
    reference = ReferenceIndex(directory)
    catalog = pd.read_parquet('data/train.parquet', columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).drop_duplicates('inchikey14')
    catalog['mass'] = catalog.molecular_formula.map(formula_mass)
    index = CandidateIndex(build_candidates(catalog, COCONUT, final=True))
    coco = pd.read_parquet(COCONUT)
    co = np.argsort(coco.exact_mass.to_numpy())
    cm = coco.exact_mass.to_numpy()[co]
    union = index.catalog.rename(columns={'inchikey14': 'inchikey', 'normalized_smiles': 'canonical_smiles', 'mass': 'exact_mass'})
    uo = np.argsort(union.exact_mass.to_numpy())
    um = union.exact_mass.to_numpy()[uo]
    old = pd.read_csv('submission_v3.csv').set_index('molecule_id').smiles
    current = pd.read_csv('kaggle_release/run_output/submission.csv').set_index('molecule_id').smiles
    lookup = dict(zip(index.catalog.inchikey14, index.catalog.normalized_smiles))
    cache, rows, predictions = {}, [], {k: [] for k in ['coconut15', 'coconut35', 'union15', 'union35']}
    for i, (mid, group) in enumerate(test.groupby('molecule_id', sort=False)):
        center, library = library_rank(group, records, matrix, masses, order)
        historical = coconut_rank(center, library, coco, cm, co, cache)
        widened = analog_rank(center, library, coco, cm, co, cache, 35, .006)
        shared15 = analog_rank(center, library, union, um, uo, cache, 15, .004)
        pool, fps = index.fps(index.query(center))
        ranked = {'coconut15': blend(library, historical),
                  'coconut35': blend(library, widened),
                  'union15': blend(library, shared15),
                  'union35': [(k, lookup[k]) for k in baseline_rank(group, pool, fps, reference, center)[:25]]}
        old_keys = [structure_key(s) for s in old[mid].split(';')]
        current_keys = [structure_key(s) for s in current[mid].split(';')]
        for variant, candidates in ranked.items():
            # Compare actual structure identity, not different canonical SMILES strings.
            keys = list(dict.fromkeys(structure_key(smi) for _, smi in candidates))
            rows.append({'molecule_id': mid, 'variant': variant, 'same_top1_historical': keys[:1] == old_keys[:1],
                         'same_order_historical': keys == old_keys, 'same_set_historical': set(keys) == set(old_keys),
                         'overlap_historical': len(set(keys) & set(old_keys)),
                         'same_order_current': keys == current_keys})
            predictions[variant].append({'molecule_id': mid, 'smiles': ';'.join(smi for _, smi in candidates)})
        if (i+1) % 100 == 0:
            print(f'visible {i+1}/400', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / 'visible_comparison.csv', index=False)
    reports = {}
    for variant, group in df.groupby('variant'):
        reports[variant] = {col: int(group[col].sum()) for col in ['same_top1_historical', 'same_order_historical', 'same_set_historical', 'same_order_current']}
        reports[variant]['mean_overlap_historical'] = float(group.overlap_historical.mean())
        pd.DataFrame(predictions[variant]).to_csv(ROOT / f'visible_{variant}.csv', index=False)
    write_json(ROOT / 'visible_reproduction.json', {'molecules': test.molecule_id.nunique(), 'variants': reports,
               'note': 'Original historical reference loader and library ranking. Comparisons are structural identity, not ground-truth accuracy. No new Kaggle submission.'})
    print(json.dumps(reports, indent=2), flush=True)
    assert reports['coconut15']['same_order_historical'] == 400, 'Historical reproduction mismatch'


if __name__ == '__main__':
    main()
