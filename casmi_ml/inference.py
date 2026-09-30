"""Offline Kaggle inference and packaging for the frozen development winner."""
import hashlib
import json
import resource
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from casmi_ml.data import features, write_json
from casmi_ml.ranking import (
    CandidateIndex,
    ReferenceIndex,
    baseline_rank,
    build_candidates,
    build_reference,
    center_mass,
    neural_rank,
    rrf,
)
from casmi_ml.training import configure, load_checkpoint


def locate(path, filename):
    path = Path(path)
    if path.exists():
        return path
    matches = list(Path('/kaggle/input').rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one mounted {filename}, found {len(matches)}')
    return matches[0]


@torch.inference_mode()
def group_probability(model, group, prep):
    values = [features(row, prep) for row in group.to_dict('records')]
    output = []
    for start in range(0, len(values), 64):
        part = values[start:start+64]
        batch = {key: torch.from_numpy(np.stack([item[i] for item in part]))
                 for i, key in enumerate(['hist', 'loss', 'meta', 'peaks', 'mask'])}
        output.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(output).mean(0)


def validate_submission(test, submission):
    expected = set(test.molecule_id)
    if set(submission.molecule_id) != expected or submission.molecule_id.duplicated().any():
        raise ValueError('Submission molecule IDs do not match the test set')
    for row in submission.itertuples():
        values = row.smiles.split(';') if isinstance(row.smiles, str) else []
        if not 1 <= len(values) <= 25:
            raise ValueError(f'Invalid candidate count: {row.molecule_id}')
        keys = []
        for smiles in values:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError(f'Invalid SMILES for {row.molecule_id}')
            keys.append(Chem.MolToInchiKey(mol)[:14])
        if len(keys) != len(set(keys)):
            raise ValueError(f'Duplicate 2D structures for {row.molecule_id}')


def predict(selection_path, data_dir, coconut_path, output):
    started = time.monotonic()
    configure()
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    train_path = locate(Path(data_dir) / 'train.parquet', 'train.parquet')
    test_path = locate(Path(data_dir) / 'test.parquet', 'test.parquet')
    coconut_path = locate(coconut_path, 'coconut_structures.parquet')
    test = pd.read_parquet(test_path)
    if selection.get('deployment_mode') == 'historical_hybrid':
        from hybrid import main as historical_hybrid
        historical_hybrid(train_path.parent, coconut_path, output)
        submission = pd.read_csv(output)
        validate_submission(test, submission)
        write_json(str(output) + '.report.json', {'molecules': len(submission),
                   'seconds': time.monotonic()-started, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'deployment_mode': 'historical_hybrid',
                   'kaggle_verified': False})
        return submission
    from baseline import formula_mass
    catalog = pd.read_parquet(train_path, columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).drop_duplicates('inchikey14')
    catalog['mass'] = catalog.molecular_formula.map(formula_mass)
    catalog = catalog[np.isfinite(catalog.mass) & (catalog.mass > 0)].copy()
    candidates = CandidateIndex(build_candidates(catalog, coconut_path, final=True))
    reference_queries = test.rename(columns={'molecule_id': 'inchikey14'})
    with tempfile.TemporaryDirectory(prefix='casmi_reference_') as reference_dir:
        build_reference(train_path, catalog, reference_queries, reference_dir, final=True)
        reference = ReferenceIndex(reference_dir)
    if not len(reference.rows):
        raise ValueError('Reference library is empty')
    models = []
    for path in selection['checkpoints']:
        path = Path(path)
        if not path.is_absolute() and (selection_path.parent / path).exists():
            path = selection_path.parent / path
        models.append(load_checkpoint(path))
    lookup = dict(zip(candidates.catalog.inchikey14, candidates.catalog.normalized_smiles))
    fallback = [(key, lookup.get(key)) for key in reference.rows.inchikey14.value_counts().head(100).index]
    output_rows, fallback_count, protected_count = [], 0, 0
    for count, (molecule_id, group) in enumerate(test.groupby('molecule_id', sort=False), 1):
        center = center_mass(group)
        pool, fps = candidates.fps(candidates.query(center))
        keys = pool.inchikey14.tolist()
        baseline = baseline_rank(group, pool, fps, reference, center)
        rankings = []
        for model, checkpoint in models:
            probability = group_probability(model, group, checkpoint['preprocessing'])
            if np.isfinite(probability).all():
                rankings.append(neural_rank(probability, keys, fps))
        ranked = rrf([baseline, rrf(rankings)], [1-selection['neural_weight'], selection['neural_weight']]) if rankings else baseline
        if selection.get('deployment_mode') == 'confidence_guarded':
            from casmi_ml.guarded import guarded_rank
            library = reference.rank(group, center)
            confidence = library[0][1] if library else 0.
            protected_count += int(confidence >= selection['confidence_threshold'])
            ranked = guarded_rank(baseline, ranked, confidence, selection['confidence_threshold'])
        if not ranked:
            fallback_count += 1
            ranked = [k for k, _ in reference.rank(group, center, fallback=True)]
        # Validate canonical keys as we emit, guarding duplicate source encodings.
        smiles_out, seen = [], set()
        for key in ranked:
            smiles = lookup.get(key)
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if mol is None:
                continue
            actual_key = Chem.MolToInchiKey(mol)[:14]
            if actual_key not in seen:
                seen.add(actual_key)
                smiles_out.append(smiles)
            if len(smiles_out) == 25:
                break
        if not smiles_out:
            for key, smiles in fallback:
                mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
                if mol is not None and mol.GetNumAtoms():
                    actual_key = Chem.MolToInchiKey(mol)[:14]
                    if actual_key not in seen:
                        smiles_out.append(smiles)
                        seen.add(actual_key)
                if len(smiles_out) == 25:
                    break
        output_rows.append({'molecule_id': molecule_id, 'smiles': ';'.join(smiles_out)})
        if count % 100 == 0:
            print(f'predicted {count}/{test.molecule_id.nunique()}', flush=True)
    submission = pd.DataFrame(output_rows)
    validate_submission(test, submission)
    submission.to_csv(output, index=False)
    write_json(str(output) + '.report.json', {'molecules': len(submission), 'seconds': time.monotonic()-started,
               'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'fallback_molecules': fallback_count, 'protected_molecules': protected_count, 'architectures': selection['architectures'],
               'neural_weight': selection['neural_weight'], 'kaggle_verified': False})
    return submission


def package(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / 'final_selection.json').read_text())
    copied = []
    for i, path in enumerate(selection['checkpoints']):
        name = f'model_{i}.pt'
        shutil.copy2(path, destination / name)
        copied.append(name)
    selection['checkpoints'] = copied
    write_json(destination / 'final_selection.json', selection)
    research_path = root / 'research_selection.json'
    if research_path.exists():
        research = json.loads(research_path.read_text())
        paths = []
        for i, path in enumerate(research['checkpoints']):
            name = f'research_model_{i}.pt'
            shutil.copy2(path, destination / name)
            paths.append(name)
        research['checkpoints'] = paths
        write_json(destination / 'research_selection.json', research)
    for filename in ['baseline.py', 'hybrid.py']:
        shutil.copy2(filename, destination / filename)
    shutil.copytree('casmi_ml', destination / 'casmi_ml', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root / 'REPORT.md', destination / 'REPORT.md')
    for filename in ['experiments.csv', 'seed_summary.json', 'dev_comparisons.json',
                     'holdout_report.json', 'diagnostic_report.json', 'manifest.json']:
        shutil.copy2(root / filename, destination / filename)
    for filename in ['input_checksums.json', 'training_data_quality.json', 'experiment_config.json', 'candidate_coverage_audit.json', 'known_spectrum_report.json', 'guarded_selection.json', 'guarded_fresh_report.json', 'guarded_dev_comparisons.json', 'fresh_holdout_manifest.json', 'delivery_verification.json', 'submission_comparison.json']:
        if (root / filename).exists():
            shutil.copy2(root / filename, destination / filename)
    shutil.copy2('external/COCONUT_ATTRIBUTION.md', destination / 'COCONUT_ATTRIBUTION.md')
    wheel_dir = Path('external/ml_wheels')
    if wheel_dir.exists():
        shutil.copytree(wheel_dir, destination / 'wheels', dirs_exist_ok=True)
    runtime = """from pathlib import Path
import sys, subprocess, json, importlib.metadata
# Attach the generated artifact directory as a private Kaggle Dataset and attach
# the competition files and the same COCONUT dataset used during validation.
roots = list(Path('/kaggle/input').rglob('final_selection.json'))
assert len(roots) == 1, f'Expected one model bundle, found {roots}'
bundle = roots[0].parent
required = json.loads((bundle / 'runtime_versions.json').read_text())
try:
    installed_rdkit = importlib.metadata.version('rdkit')
except importlib.metadata.PackageNotFoundError:
    installed_rdkit = None
if installed_rdkit != required['rdkit']:
    wheels = list((bundle / 'wheels').glob('rdkit*.whl'))
    assert len(wheels) == 1, 'Bundle must include the training-version RDKit wheel'
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheels[0])])
import torch
sys.path.insert(0, str(bundle))
from casmi_ml.inference import predict
predict(bundle / 'final_selection.json', '/kaggle/input/competition',
        '/kaggle/input/coconut_structures.parquet', '/kaggle/working/submission.csv')
"""
    notebook = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# CASMI frozen local-validation winner\n',
        'Offline inference. Attach the generated bundle, competition and COCONUT. The bundle supplies the RDKit wheel.\n',
        'Local validation results are in REPORT.md. Latest competition rules require manual verification.']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': runtime.splitlines(True)}],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4, 'nbformat_minor': 5}
    write_json(destination / 'casmi_winner.ipynb', notebook)
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'pandas', 'pyarrow', 'scipy', 'rdkit']}
    write_json(destination / 'runtime_versions.json', versions)
    (destination / 'README.md').write_text('''# Kaggle bundle

This is a local bundle, not an uploaded Kaggle Dataset or a verified Kaggle run.
Upload this directory as a private dataset, attach it to casmi_winner.ipynb together
with the competition data and COCONUT structures. The bundle includes the training
RDKit wheel for CPython 3.12, Linux x86_64 (manylinux_2_28).
Keep internet disabled and GPU disabled. PyTorch must be present in the Kaggle image;
exact local versions are recorded in runtime_versions.json.
The notebook finds mounted files and writes /kaggle/working/submission.csv.
Read REPORT.md for measured local performance. Verify current competition rules
and runtime limits before submitting. No upload or submission is automated.
''')
    checksums = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS.json' and '__pycache__' not in path.parts:
            checksums[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(destination / 'SHA256SUMS.json', checksums)
    print(f'Packaged {destination}', flush=True)
