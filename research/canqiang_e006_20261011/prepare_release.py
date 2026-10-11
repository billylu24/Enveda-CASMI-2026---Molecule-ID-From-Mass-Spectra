"""Preserve canqiang E006 V1 code and append read-only execution acceptance."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVID = ROOT / 'evidence'
SOURCE = 'canqiang/casmi26-e006-window-40-ice-fallback-true'
KERNEL = 'giaok246/casmi26-canqiang-e006-reproduction'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def code(nb):
    return [''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code']

source = next((ROOT / 'source').glob('*.ipynb'))
original = json.loads(source.read_text())
assert json.loads((EVID / 'source_api_metadata.json').read_text())['currentVersionNumber'] == 1
assert code(original) == code(json.loads((EVID / 'api_source.ipynb').read_text()))
for text in code(original):
    ast.parse(text)
nb = copy.deepcopy(original)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
nb['cells'].insert(0, dict(cell_type='markdown', metadata={}, source='''# Canqiang E006 reproduction

Source: canqiang/casmi26-e006-window-40-ice-fallback-true, public V1, retrieved October 11, 2026. All 30 original code cells and upstream credits are retained without modification, including E004 observers and the E006 overlay (window=40, ICE fallback=True). Original T4 image and 16 dataset versions are retained. This private account copy adds only a read-only acceptance cell. The original 12-molecule preview switch and complete hidden-test code submission remain in place. Visible output contains placeholders and is never submitted directly. Results are reported from this reproduction's official Kaggle submission, without treating inherited source score claims as our score.
'''))
checks = '''# Read-only final acceptance; source inference and ranking are unchanged.
import hashlib as _accept_hash
from pathlib import Path as _AcceptPath
_accept_frame = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
_accept_ids = set(E004_PREFLIGHT['input_ids'])
_accept_processed = set(te.molecule_id.astype(str))
_accept_expected = _accept_ids if IS_RERUN else set(sorted(_accept_ids)[:SMOKE_N])
assert _accept_processed == _accept_expected
assert list(_accept_frame.columns) == ['molecule_id', 'smiles']
assert _accept_frame.molecule_id.is_unique and set(_accept_frame.molecule_id) == _accept_ids
_accept_sample = pd.read_csv(os.path.join(E004_PREFLIGHT['runtime_input_path'], 'sample_submission.csv'), dtype=str)
assert list(_accept_frame.molecule_id) == list(_accept_sample.molecule_id)
assert n_err == 0 and set(map(str, BASE)) == _accept_processed
assert _accept_processed <= set(map(str, ENG))
assert _accept_frame.smiles.str.split(';').map(len).between(1, 25).all()
for _accept_row in _accept_frame.itertuples(index=False):
    if _accept_row.molecule_id in _accept_processed:
        _accept_keys = [metric_identity(s) for s in _accept_row.smiles.split(';')]
        assert None not in _accept_keys and len(_accept_keys) == len(set(_accept_keys))
assert FACTORIAL_REPORT['experiment'] == 'E006'
assert FACTORIAL_REPORT['window'] == 40 and FACTORIAL_REPORT['ice_fallback'] is True
assert FACTORIAL_REPORT['same_run_control_reconstructed']
assert FACTORIAL_REPORT['submission_sha256'] == _accept_hash.sha256(_AcceptPath('submission.csv').read_bytes()).hexdigest()
if IS_RERUN:
    assert _accept_processed == _accept_ids, 'Unprocessed hidden-test molecules'
_accept_report = dict(passed=True, source='canqiang/casmi26-e006-window-40-ice-fallback-true',
    source_version=1, original_code_cells=30, hidden_rerun=bool(IS_RERUN),
    rows=len(_accept_frame), processed=len(_accept_processed), preview_placeholders=len(_accept_ids-_accept_processed),
    base_errors=int(n_err), main_rdkit=rdkit.__version__,
    window=40, ice_fallback=True, same_run_control_reconstructed=True,
    submission_sha256=FACTORIAL_REPORT['submission_sha256'],
    factorial=FACTORIAL_REPORT, stages=E004_REPORT['stages'],
    runtime_seconds=round(time.time()-T0, 2))
_AcceptPath('reproduction_acceptance.json').write_text(json.dumps(_accept_report, indent=2, default=str))
print('REPRODUCTION_ACCEPTANCE', json.dumps(_accept_report, default=str))
'''
ast.parse(checks)
nb['cells'].append(dict(cell_type='code', metadata={}, source=checks, outputs=[], execution_count=None))
assert code(nb)[:-1] == code(original)
release = ROOT / 'release'
release.mkdir(exist_ok=True)
(release / 'reproduction.ipynb').write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / 'source/kernel-metadata.json').read_text())
meta.pop('id_no', None)
meta.update(id=KERNEL, title='CASMI26 Canqiang E006 Reproduction', code_file='reproduction.ipynb', is_private=True)
assets = json.loads((EVID / 'dependencies.json').read_text())
assert len(assets) == 16
meta['dataset_sources'] = [f"{x['ref']}/{x['version']}" for x in assets]
(release / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
identity = dict(source=SOURCE, source_version=1, original_code_cells=30, source_sha256=sha(source),
    source_code_sha256=hashlib.sha256('\n'.join(code(original)).encode()).hexdigest(),
    original_code_exact=True, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    changes=['Added attribution markdown and read-only final acceptance cell; all original code unchanged.',
             'Private account copy; pin dataset versions fetched before publication.'],
    release_files={p.name: sha(p) for p in release.iterdir() if p.is_file()})
(EVID / 'release_identity.json').write_text(json.dumps(identity, indent=2))
print(json.dumps(dict(kernel=KERNEL, original_code_cells=30, cells=len(nb['cells']), datasets=len(assets), release_files=identity['release_files']), indent=2))
