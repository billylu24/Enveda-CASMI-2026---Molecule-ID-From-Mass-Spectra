"""Prepare a source-exact V30 BIO-union reproduction for Kaggle."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = 'flexonafft/casmi26-closing-the-evidence-gap'
KERNEL = 'giaok246/casmi26-flexonafft-v30-bio-union-reproduction'
VERSION = 30

def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = ROOT / 'source/casmi26-closing-the-evidence-gap.ipynb'
    nb = json.loads(source.read_text())
    snapshot = json.loads((ROOT / 'evidence/source_snapshot.json').read_text())
    meta = snapshot['metadata']
    assert meta['currentVersionNumber'] == VERSION
    assert sha(source) == snapshot['source_sha256']
    assert len(code(nb)) == 6
    for s in code(nb): ast.parse(s)
    release = copy.deepcopy(nb)
    for cell in release['cells']:
        if cell['cell_type'] == 'code': cell.update(outputs=[], execution_count=None)
    release['cells'].insert(0, dict(cell_type='markdown', metadata={}, source='''# Closing the Evidence Gap — public V30 reproduction

Original notebook: Igor Zharov / @flexonafft, https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap, public version 30, retrieved October 17, 2026 (America/Los_Angeles). The source header calls the underlying recipe V22 BIO Structure Union; public notebook version 30 and recipe V22 are different identifiers.

All six original code cells, embedded modules, and inherited credits are preserved unchanged. This copy adds attribution and a read-only final verification. It combines sergiomalv BIO structures with additional prvsiyan structures, recalculating added fingerprints with engine2. The original pinned Python/T4 image, offline runtime, and 16 dataset inputs are retained. Dataset versions are chosen from Ready releases before upstream lastRunTime; original attachment versions are unavailable from the API.

The visible run processes 12 molecules and fills remaining sample rows with placeholders. This is an execution and source-parity check, not accuracy validation. The inherited 0.451 claim refers to the underlying recipe, not a measured score for this BIO-union reproduction. No competition submission is performed by this notebook publisher. Dataset/component licenses remain independent of the notebook license; keep all original attributions.
'''))
    checks = '''# Read-only reproduction verification; inference and candidate order are unchanged.
import importlib.metadata as _versions_api
import sys as _sys
_scope = experiment.scope
_ids = set(map(str, _scope['E004_PREFLIGHT']['input_ids']))
_active = set(map(str, _scope['te'].molecule_id))
_expected = _ids if _scope['IS_RERUN'] else set(sorted(_ids)[:_scope['SMOKE_N']])
assert _active == _expected
_frame = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
_sample = pd.read_csv(Path(_scope['E004_PREFLIGHT']['runtime_input_path']) / 'sample_submission.csv', dtype=str)
assert _frame.columns.tolist() == ['molecule_id', 'smiles']
assert _frame.molecule_id.is_unique and _frame.molecule_id.tolist() == _sample.molecule_id.tolist()
assert _frame.smiles.map(lambda s: bool(s.strip()) and 1 <= len(s.split(';')) <= 25).all()
assert set(map(str, _scope['BASE'])) == _active and _active <= set(map(str, _scope['ENG']))
assert _scope['n_err'] == 0
_bio = json.loads(Path('eng/bio_union_audit.json').read_text())
_e2 = json.loads(Path('e2r2_stats.json').read_text())
assert _bio['new_rows'] > 0 and _bio['old_rows'] > 0
assert _bio['added_rows'] + _bio['overlap_rows'] + _bio['invalid_rows'] == _bio['old_rows']
assert 'casmi-chebi-lmsd-structures' in _bio['new_directory']
assert 'chebi-lipidmaps-casmi26' in _bio['old_meta_path']
assert _e2['status'] == 'applied'
assert result['mode'] == 'bio_union' and result['changed_rows_vs_same_run_source_output'] == 0
assert result['submission_sha256'] == hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()
assert Path('submission.csv').read_bytes() == Path('submission_v22_control.csv').read_bytes()
_packages = {}
for _pkg in ['numpy','pandas','scipy','scikit-learn','numba','torch','rdkit','lightgbm']:
    try: _packages[_pkg] = _versions_api.version(_pkg)
    except _versions_api.PackageNotFoundError: pass
_report = dict(passed=True,source='flexonafft/casmi26-closing-the-evidence-gap',source_version=30,
    recipe='V22 BIO Structure Union',original_code_cells=6,hidden_rerun=bool(_scope['IS_RERUN']),
    rows=len(_frame),processed=len(_active),placeholders=len(_ids-_active),processed_ids=sorted(_active),
    base_errors=int(_scope['n_err']),python=_sys.version,packages=_packages,bio_union=_bio,e2r2=_e2,
    adaptation=result,simulators=_scope['E004_REPORT']['stages'],
    submission_sha256=result['submission_sha256'],measured_score=None,
    runtime_seconds=round(time.time()-_scope['T0'],2))
_frame[_frame.molecule_id.isin(_active)].to_csv('processed_predictions.csv',index=False)
Path('reproduction_acceptance.json').write_text(json.dumps(_report,indent=2,default=str))
print('REPRODUCTION_ACCEPTANCE',json.dumps(_report,default=str))
'''
    ast.parse(checks)
    release['cells'].append(dict(cell_type='code',metadata={},source=checks,outputs=[],execution_count=None))
    assert code(release)[:-1] == code(nb)
    dest = ROOT / 'release'; dest.mkdir(exist_ok=True)
    (dest/'reproduction.ipynb').write_text(json.dumps(release,indent=1))
    deps = json.loads((ROOT/'evidence/dependencies.json').read_text())
    assert {d['ref'] for d in deps} == set(meta['datasetDataSources']) and len(deps) == 16
    attached = dict(id=KERNEL,title='CASMI26 Flexonafft V30 BIO Union Reproduction',code_file='reproduction.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
        enable_internet=False,keywords=['gpu'],dataset_sources=[f"{d['ref']}/{d['version']}" for d in deps],
        kernel_sources=[],competition_sources=meta['competitionDataSources'],model_sources=[],
        docker_image=meta['dockerImage'],machine_shape=meta['machineShape'])
    (dest/'kernel-metadata.json').write_text(json.dumps(attached,indent=2))
    identity=dict(source=SOURCE,source_version=VERSION,source_sha256=sha(source),
        original_code_cells=6,original_code_exact=True,
        source_code_sha256=hashlib.sha256('\n'.join(code(nb)).encode()).hexdigest(),kernel=KERNEL,
        release_files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},
        changes=['Attribution markdown','Read-only acceptance appended; all original code unchanged'])
    (ROOT/'evidence/release_identity.json').write_text(json.dumps(identity,indent=2))
    print(json.dumps(identity,indent=2))

if __name__ == '__main__': main()
