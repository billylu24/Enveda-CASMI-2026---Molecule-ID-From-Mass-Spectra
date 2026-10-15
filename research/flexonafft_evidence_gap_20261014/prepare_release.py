"""Build a source-exact Iteration 07 Kaggle reproduction with read-only acceptance."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVID = ROOT / 'evidence'
KERNEL = 'giaok246/casmi26-flexonafft-evidence-gap-reproduction'
SOURCE = 'flexonafft/casmi26-closing-the-evidence-gap'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

def main():
    source = ROOT / 'source/casmi26-closing-the-evidence-gap.ipynb'
    nb = json.loads(source.read_text())
    meta = json.loads((EVID / 'source_api_metadata.json').read_text())
    assert meta['currentVersionNumber'] == 15
    assert code(nb) == code(json.loads((EVID / 'api_source.ipynb').read_text()))
    assert len(code(nb)) == 9
    for text in code(nb): ast.parse(text)
    # Readable generated mirrors; execution always uses the original notebook cells.
    for i, text in enumerate(code(nb)):
        (ROOT / 'source' / f'cell_{i:02}.py').write_text(text)
    embedded = ast.literal_eval(next(node.value for node in ast.parse(code(nb)[5]).body
                                    if isinstance(node, ast.Assign)))
    mirror = ROOT / 'source' / 'embedded'; mirror.mkdir(exist_ok=True)
    for i, text in embedded.items():
        ast.parse(text)
        (mirror / f'cell_{i:02}.py').write_text(text)
    release = copy.deepcopy(nb)
    for cell in release['cells']:
        if cell['cell_type'] == 'code': cell.update(outputs=[], execution_count=None)
    release['cells'].insert(0, dict(cell_type='markdown', metadata={}, source='''# Flexonafft: Closing the Evidence Gap — V15 reproduction

Original author: Igor Zharov / @flexonafft. Source: https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap, public version 15, retrieved October 14, 2026 (America/Los_Angeles). All nine original code cells, embedded source, and upstream credits are retained unchanged. The final algorithm is Iteration 07, replicated fragment evidence with a median over at least three distinct informative spectra. This copy adds only this attribution and a read-only acceptance cell. The original Python/T4 container and 15 version-frozen public dataset inputs are retained; versions are selected from ready releases before the upstream lastRunTime because the API does not disclose original attachment versions. Main RDKit 2026.03.3 and simulator RDKit 2025.03.6 remain separate.

The original visible commit runs 12 molecules; all remaining rows are placeholders. The hidden-test rerun code is retained. A visible smoke result is an execution and source-parity check, not an accuracy estimate. The inherited 0.441 claim is not this reproduction's measured score. This task does not automatically submit to the competition.
'''))
    checks = '''# Read-only acceptance: do not change inference, routing, or prediction order.
import importlib.metadata as _accept_metadata
_scope = experiment.scope
_frame = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
_ids = set(_scope['E004_PREFLIGHT']['input_ids'])
_active = set(_scope['te'].molecule_id.astype(str))
_expected = _ids if _scope['IS_RERUN'] else set(sorted(_ids)[:_scope['SMOKE_N']])
assert _active == _expected
assert list(_frame.columns) == ['molecule_id', 'smiles']
assert _frame.molecule_id.is_unique and set(_frame.molecule_id) == _ids
_sample = pd.read_csv(Path(_scope['E004_PREFLIGHT']['runtime_input_path']) / 'sample_submission.csv', dtype=str)
assert list(_frame.molecule_id) == list(_sample.molecule_id)
assert _scope['n_err'] == 0
assert set(map(str, _scope['BASE'])) == _active
assert _active <= set(map(str, _scope['ENG']))
_control = pd.read_csv('submission_0441_control.csv', dtype=str, keep_default_na=False)
SubmissionBuilder(_scope).validate(_frame, _control, _active)
assert result['source_control_reconstructed'] and result['changed_rows'] == 0
assert not result['fragment_errors']
assert not iteration_result['details']['errors']
assert iteration_result['experiment'] == 'replicated_fragment_fallback'
assert iteration_result['details']['aggregation'] == 'median'
assert iteration_result['details']['min_distinct_informative_spectra'] == 3
assert iteration_result['details']['max_group_size'] == 8
assert iteration_result['details']['max_bonds'] == 40
assert iteration_result['details']['contrast_lambda'] == 1.25
assert iteration_result['submission_sha256'] == hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()
assert iteration_result['control_sha256'] == hashlib.sha256(Path('submission_0441_control.csv').read_bytes()).hexdigest()
_versions = {}
for _pkg in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'numba', 'torch', 'rdkit', 'lightgbm']:
    try: _versions[_pkg] = _accept_metadata.version(_pkg)
    except _accept_metadata.PackageNotFoundError: pass
_accept = dict(passed=True, source='flexonafft/casmi26-closing-the-evidence-gap', source_version=15,
    original_code_cells=9, rows=len(_frame), processed=len(_active), placeholders=len(_ids-_active),
    hidden_rerun=bool(_scope['IS_RERUN']), python=__import__('sys').version, packages=_versions,
    main_rdkit=_scope['rdkit'].__version__, base_errors=int(_scope['n_err']),
    initial01=result, iteration07=iteration_result, simulators=_scope['E004_REPORT']['stages'],
    submission_sha256=iteration_result['submission_sha256'],
    measured_score=None, runtime_seconds=round(time.time()-_scope['T0'], 2))
Path('reproduction_acceptance.json').write_text(json.dumps(_accept, indent=2, default=str))
print('REPRODUCTION_ACCEPTANCE', json.dumps(_accept, default=str))
'''
    ast.parse(checks)
    release['cells'].append(dict(cell_type='code', metadata={}, source=checks, outputs=[], execution_count=None))
    assert code(release)[:-1] == code(nb)
    dest = ROOT / 'release'; dest.mkdir(exist_ok=True)
    (dest / 'reproduction.ipynb').write_text(json.dumps(release, indent=1))
    attached = json.loads((ROOT / 'source/kernel-metadata.json').read_text())
    attached.pop('id_no', None)
    attached.update(id=KERNEL, title='CASMI26 Flexonafft Evidence Gap Reproduction', code_file='reproduction.ipynb', is_private=True)
    deps = json.loads((EVID / 'dependencies.json').read_text()); assert len(deps) == 15
    attached['dataset_sources'] = [f"{x['ref']}/{x['version']}" for x in deps]
    (dest / 'kernel-metadata.json').write_text(json.dumps(attached, indent=2))
    identity = dict(source=SOURCE, source_version=15, original_code_cells=9,
        source_sha256=sha(source), original_code_exact=True,
        source_code_sha256=hashlib.sha256('\n'.join(code(nb)).encode()).hexdigest(),
        prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        changes=['Added attribution markdown and read-only acceptance; all original code unchanged.',
                 'Private account copy; freeze 15 dependency versions before source lastRunTime.'],
        release_files={p.name: sha(p) for p in dest.iterdir() if p.is_file()})
    (EVID / 'release_identity.json').write_text(json.dumps(identity, indent=2))
    print(json.dumps(identity, indent=2))

if __name__ == '__main__': main()
