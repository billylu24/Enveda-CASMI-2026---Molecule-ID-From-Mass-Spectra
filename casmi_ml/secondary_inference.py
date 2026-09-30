"""Offline deployment of the development-frozen, unknown-spectrum-first recipe."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from baseline import ADDUCT_MASS, formula_mass, load_candidates, make_matrix
from casmi_ml.data import write_json
from casmi_ml.inference import group_probability, locate, validate_submission
from casmi_ml.ranking import (
    CandidateIndex,
    ReferenceIndex,
    baseline_rank,
    build_candidates,
    neural_rank,
    rrf,
)
from casmi_ml.training import configure, load_checkpoint
from hybrid import blend, coconut_rank, library_rank


class MemoryReference(ReferenceIndex):
    def __init__(self, records, matrix):
        self.rows = pd.DataFrame(records, columns=['mass', 'inchikey14', 'normalized_smiles'])
        self.matrix = matrix
        self.order = np.argsort(self.rows.mass.to_numpy())
        self.mass = self.rows.mass.to_numpy()[self.order]


def low_confidence_rank(historical, current, neural, weight):
    extended = historical + [k for k in current if k not in set(historical)]
    return rrf([extended, neural], [1-weight, weight])


def load_deployment_checkpoint(path, family='fingerprint'):
    if family == 'fingerprint':
        return load_checkpoint(path)
    if family != 'scale':
        raise ValueError(f'Unsupported encoder family {family}')
    from casmi_ml.scale_models import ScaleModel
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ScaleModel(checkpoint['architecture'], checkpoint['metadata_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint


def predict(recipe_path, data_dir, coconut_path, output):
    started = time.monotonic()
    configure(threads=4)
    recipe_path = Path(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    if sum(name in recipe for name in ['router', 'direct_ranker', 'candidate_expansion']) > 1:
        raise ValueError('Combining experimental extensions has not been validated')
    config = recipe['config']
    if config['kind'] != 'free_top1' or config['base'] != 'coconut15':
        raise ValueError('This deployment implements the frozen historical/free-top1 recipe only')
    checkpoint_path = Path(recipe['checkpoint'])
    if not checkpoint_path.is_absolute() and (recipe_path.parent / checkpoint_path).exists():
        checkpoint_path = recipe_path.parent / checkpoint_path
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != recipe['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    external = None
    if 'candidate_expansion' in recipe:
        expansion_spec = recipe['candidate_expansion']
        external_path = Path(expansion_spec['path'])
        if not external_path.is_absolute() and (recipe_path.parent / external_path).exists():
            external_path = recipe_path.parent / external_path
        if hashlib.sha256(external_path.read_bytes()).hexdigest() != expansion_spec['sha256']:
            raise ValueError('External candidate checksum mismatch')
        external = pd.read_parquet(external_path)
    direct = None
    if 'direct_ranker' in recipe:
        from casmi_ml.direct_models import DirectRanker
        direct_spec = recipe['direct_ranker']
        direct_path = Path(direct_spec['path'])
        if not direct_path.is_absolute() and (recipe_path.parent / direct_path).exists():
            direct_path = recipe_path.parent / direct_path
        if hashlib.sha256(direct_path.read_bytes()).hexdigest() != direct_spec['sha256']:
            raise ValueError('Direct ranker checksum mismatch')
        direct_checkpoint = torch.load(direct_path, map_location='cpu', weights_only=True)
        if direct_checkpoint['encoder_sha256'] != recipe['checkpoint_sha256']:
            raise ValueError('Direct ranker was trained with a different spectrum encoder')
        direct = DirectRanker(direct_checkpoint['architecture'])
        direct.load_state_dict(direct_checkpoint['state_dict'])
        direct.eval()
    router = None
    if 'router' in recipe:
        import joblib
        router_spec = recipe['router']
        router_path = Path(router_spec['path'])
        if not router_path.is_absolute() and (recipe_path.parent / router_path).exists():
            router_path = recipe_path.parent / router_path
        if hashlib.sha256(router_path.read_bytes()).hexdigest() != router_spec['sha256']:
            raise ValueError('Router checksum mismatch')
        if router_spec['high'] != 'H' or router_spec['low'] != 'F':
            raise ValueError('Deployment supports calibrated H/F routing only')
        router = joblib.load(router_path)
    train_path = locate(Path(data_dir) / 'train.parquet', 'train.parquet')
    test_path = locate(Path(data_dir) / 'test.parquet', 'test.parquet')
    coconut_path = locate(coconut_path, 'coconut_structures.parquet')
    test = pd.read_parquet(test_path)
    coconut = pd.read_parquet(coconut_path, columns=['inchikey', 'canonical_smiles', 'exact_mass'])
    neutral = test.precursor_mz.to_numpy() - test.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral)]
    if not len(neutral):
        raise ValueError('No supported precursor masses in the supplied test set')
    records, _ = load_candidates(train_path, neutral)
    if not records:
        raise ValueError('No reference spectra retained')
    matrix = make_matrix([r[3] for r in records])
    records = [r[:3] for r in records]
    masses = np.asarray([r[0] for r in records])
    order = np.argsort(masses)
    coco_order = np.argsort(coconut.exact_mass.to_numpy())
    coco_masses = coconut.exact_mass.to_numpy()[coco_order]
    model = checkpoint = candidates = reference = lookup = expanded = None
    cache, rows, audit = {}, [], []
    for molecule_id, group in test.groupby('molecule_id', sort=False):
        center, library = library_rank(group, records, matrix, masses, order)
        analog = coconut_rank(center, library, coconut, coco_masses, coco_order, cache)
        historical = blend(library, analog)
        confidence = library[0][2] if library else 0.
        protected = router is None and confidence >= config['threshold']
        reliability = None
        if protected:
            output_pairs = historical
        else:
            if model is None:
                model, checkpoint = load_deployment_checkpoint(checkpoint_path, recipe.get('encoder_family', 'fingerprint'))
                catalog = pd.read_parquet(train_path, columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).drop_duplicates('inchikey14')
                catalog['mass'] = catalog.molecular_formula.map(formula_mass)
                candidates = CandidateIndex(build_candidates(catalog, coconut_path, final=True))
                if external is not None:
                    from casmi_ml.candidate_catalog import expanded_pool
                    expanded = CandidateIndex(expanded_pool(candidates.catalog, external))
                reference = MemoryReference(records, matrix)
                structures_catalog = expanded.catalog if expanded is not None else candidates.catalog
                lookup = dict(zip(structures_catalog.inchikey14, structures_catalog.normalized_smiles))
            pool, fps = candidates.fps(candidates.query(center))
            current = baseline_rank(group, pool, fps, reference, center)
            probability = group_probability(model, group, checkpoint['preprocessing'])
            if not np.isfinite(probability).all():
                raise ValueError(f'Nonfinite neural output for {molecule_id}')
            neural = neural_rank(probability, pool.inchikey14.tolist(), fps)
            if direct is not None:
                from casmi_ml.direct_models import score_group
                scores = score_group(direct, model, group, checkpoint['preprocessing'], pool, fps)
                keys = pool.inchikey14.tolist()
                direct_ranking = [keys[j] for j in sorted(range(len(keys)), key=lambda j: (-scores[j], keys[j]))]
                neural = rrf([neural, direct_ranking], [1-direct_spec['weight'], direct_spec['weight']])
            ranking = low_confidence_rank([k for k, _ in historical], current, neural, config['weight'])
            if expanded is not None:
                new_pool, new_fps = expanded.fps(expanded.query(center))
                new_current = baseline_rank(group, new_pool, new_fps, reference, center)
                new_neural = neural_rank(probability, new_pool.inchikey14.tolist(), new_fps)
                new_ranking = low_confidence_rank([k for k, _ in historical], new_current, new_neural, config['weight'])
                ranking = rrf([ranking, new_ranking], [1-expansion_spec['weight'], expansion_spec['weight']])
            if router is not None:
                from casmi_ml.router_features import router_features
                margin = confidence - library[1][2] if len(library) > 1 else confidence
                features = router_features(group, probability, pool.inchikey14.tolist(), fps,
                                           [k for k, _ in historical], current, confidence, margin, center)
                reliability = float(router.predict_proba([features])[0, 1])
                protected = reliability >= router_spec['threshold']
            # Preserve the original structure representation for historical entries.
            structures = {**lookup, **dict(historical)}
            output_pairs = [(k, structures[k]) for k in ranking]
            if protected:
                output_pairs = historical
        smiles, seen = [], set()
        for _, smi in output_pairs:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not mol.GetNumAtoms():
                continue
            key = Chem.MolToInchiKey(mol)[:14]
            if key not in seen:
                smiles.append(smi)
                seen.add(key)
            if len(smiles) == 25:
                break
        if not smiles:
            raise ValueError(f'No valid predictions for {molecule_id}')
        rows.append({'molecule_id': molecule_id, 'smiles': ';'.join(smiles)})
        audit.append({'molecule_id': molecule_id, 'confidence': confidence,
                      'protected': protected, 'candidates': len(smiles),
                      'router_reliability': reliability})
        if len(rows) % 100 == 0:
            print(f'predicted {len(rows)}/{test.molecule_id.nunique()}', flush=True)
    submission = pd.DataFrame(rows)
    validate_submission(test, submission)
    submission.to_csv(output, index=False)
    pd.DataFrame(audit).to_csv(str(output) + '.routing.csv', index=False)
    write_json(str(output) + '.report.json', {
        'molecules': len(rows), 'protected_molecules': sum(r['protected'] for r in audit),
        'neural_molecules': sum(not r['protected'] for r in audit), 'config': config,
        'checkpoint_sha256': recipe['checkpoint_sha256'],
        'router': recipe.get('router'),
        'direct_ranker': recipe.get('direct_ranker'),
        'candidate_expansion': recipe.get('candidate_expansion'),
        'seconds': time.monotonic()-started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'note': 'Visible-output consistency is not evidence of leaderboard improvement.',
    })
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default='artifacts/ablation_20260928/experimental_recipe.json')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--coconut', default='external/coconut_structures.parquet')
    parser.add_argument('--output', default='submission_secondary.csv')
    args = parser.parse_args()
    predict(args.recipe, args.data_dir, args.coconut, args.output)
