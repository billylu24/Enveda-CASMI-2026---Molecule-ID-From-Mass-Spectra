"""Freeze the requested public notebook and add full execution acceptance checks."""
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_REF = 'samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438'
KERNEL = 'giaok246/casmi26-samanyu-iceberg-0-438-reproduction'
source = next((ROOT / 'source').glob('*.ipynb'))
original = json.loads(source.read_text())
nb = copy.deepcopy(original)
metadata = json.loads((ROOT / 'evidence/source_api_metadata.json').read_text())
assert metadata['currentVersionNumber'] == 1
api_nb = json.loads((ROOT / 'evidence/api_source.ipynb').read_text())
code = lambda n: [''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code']
assert code(original) == code(api_nb)
# Preserve the original public12 smoke switch and full hidden-test budgets.
checks = '''# Verify full execution; no changes to models, candidates, ranking or fusion.
assert dev == 'cuda' and torch.cuda.is_available(), 'Source GPU required'
assert rdkit.__version__ == '2026.03.3'
assert n_err == 0
_active_ids = set(te.molecule_id.astype(str))
assert set(map(str, BASE)) == _active_ids
assert set(map(str, ENG)) == _active_ids, 'Incomplete second-engine execution'
assert set(map(str, PC)) == _active_ids, 'Incomplete PubChem execution'
_final_check = pd.read_csv('submission.csv')
_sample_check = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
assert list(_final_check.columns) == ['molecule_id', 'smiles']
assert _final_check.molecule_id.is_unique
assert list(_final_check.molecule_id) == list(_sample_check.molecule_id)
_expected_active = set(_sample_check.molecule_id.astype(str)) if IS_RERUN else set(sorted(_sample_check.molecule_id.astype(str))[:SMOKE_N])
assert _active_ids == _expected_active, 'Unprocessed active molecule'
_processed_check = _final_check[_final_check.molecule_id.astype(str).isin(_active_ids)]
assert _final_check.smiles.notna().all()
_counts_check = _processed_check.smiles.str.split(';').map(len)
assert _counts_check.between(1, 25).all()
for _r in _processed_check.itertuples():
    _smis = _r.smiles.split(';')
    assert all(Chem.MolFromSmiles(_s) is not None for _s in _smis), _r.molecule_id
    assert len(first_ranks(_smis)) == len(_smis), _r.molecule_id
assert _summary['active_molecules'] == len(_active_ids)
assert _summary['output_rows'] == len(_final_check)
if IS_RERUN:
    assert len(_active_ids) == len(_final_check)
assert _summary['monotone_rank_assertions_passed']
_health_check = {}
for _name, _stage in _stage_meta.items():
    assert _stage['status'] == 'ok', (_name, _stage)
    assert _stage['n_mols_scored'] == _stage['n_mols_covered'] > 0, (_name, _stage)
    assert _stage.get('n_pred_fail', 0) == 0 and _stage.get('n_feat_fail', 0) == 0
    assert _stage['device'] == 'cuda' and _stage['rdkit'] == '2025.03.6'
    _allowed_errors = []
    for _err in _stage.get('errors', []):
        _match = re.fullmatch(r'smiles rejected \\((inchi round trip|elements \\[[^\\]]*\\]|element outside VALID_ATOM_NUM|> 100 heavy atoms)\\): (.+)', _err)
        assert _match is not None, (_name, _err)
        _allowed_errors.append(dict(reason=_match.group(1), smiles=_match.group(2)))
    assert _stage.get('n_bad_smiles', 0) == len(_allowed_errors)
    _health_check[_name] = dict(status='ok', scored=_stage['n_mols_scored'], covered=_stage['n_mols_covered'], unsupported=_allowed_errors)
_acceptance = dict(passed=True, source='samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438', source_version=1,
    rows=len(_final_check), processed=len(_active_ids), hidden_rerun=IS_RERUN, preview_placeholders=len(_final_check)-len(_active_ids), base_errors=n_err,
    engine2_molecules=len(ENG), pubchem_molecules=len(PC),
    candidate_min=int(_counts_check.min()), candidate_max=int(_counts_check.max()),
    rdkit=rdkit.__version__, gpu=torch.cuda.get_device_name(0), stages=_health_check,
    ice_only_fusion_molecules=n_ice_only, fusion_window=FUSE_N,
    runtime_seconds=round(time.time()-T0, 2),
    submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
    visible_data_note='Visible test overlaps training; execution acceptance is not a generalization estimate.')
Path('reproduction_acceptance.json').write_text(json.dumps(_acceptance, indent=2))
print(json.dumps(_acceptance, indent=2))
'''
ast.parse(checks)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        if True:
            assert ''.join(cell['source']) == ''.join(original['cells'][i]['source'])
        cell['outputs'] = []
        cell['execution_count'] = None
nb['cells'].append(dict(cell_type='code', metadata={}, source=checks, outputs=[], execution_count=None))
nb['cells'].insert(0, dict(cell_type='markdown', metadata={}, source='''# Samanyu ICEBERG / GLACIER reproduction

Source: samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438, public V1, retrieved October 10, 2026. All source credits retained. The title's 0.438 is an upstream claim, not our measured score. Original model, weights, thresholds, 80-candidate fusion window, metric-safe final tail, and original hidden-test inference are retained. Original 12-molecule commit smoke and full hidden rerun are preserved. Added assertions verify all active molecules and stage completion; hidden rerun rejects unprocessed rows. Public data overlaps training; no generalization score is claimed from visible data.
'''))
release = ROOT / 'release'
release.mkdir(exist_ok=True)
(release / 'reproduction.ipynb').write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / 'source/kernel-metadata.json').read_text())
meta.pop('id_no', None)
meta.update(id=KERNEL, title='CASMI26 Samanyu ICEBERG 0.438 Reproduction', code_file='reproduction.ipynb', is_private=True)
assets = json.loads((ROOT / 'evidence/dependencies.json').read_text())
meta['dataset_sources'] = [f"{x['ref']}/{x['version']}" for x in assets]
(release / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
identity = dict(source=SOURCE_REF, source_version=1, retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_api_sha256=hashlib.sha256((ROOT/'evidence/api_source.ipynb').read_bytes()).hexdigest(),
    changes=[dict(cell='added', reason='Original code cells are identical. Verify all active molecules, stage completion, output validity and metric identity uniqueness; hidden rerun rejects unprocessed rows.')],
    metadata_changes=['private account copy', 'freeze currently attached public dataset versions'],
    release_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in release.iterdir() if p.is_file()})
(ROOT / 'evidence/release_identity.json').write_text(json.dumps(identity, indent=2))
print(json.dumps(dict(kernel=KERNEL, cells=len(nb['cells']), inputs=len(assets), identity=identity['release_files']), indent=2))
