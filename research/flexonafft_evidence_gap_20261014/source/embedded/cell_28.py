METRIC_SOURCE_SHA256 = 'c61d163f1bb768313fd0863822952f035f343924f37e3f5e4b5883ccef7e6201'
'''Final pass: no change to any V1 model, fusion weight, promotion gate or simulator.'''
import collections
import hashlib
from pathlib import Path
assert rdkit.__version__ == '2026.03.3'
assert n_err == 0, f'Base inference errors: {n_err}'
_before = pd.read_csv('submission.csv')
_before.to_csv('baseline_submission.csv', index=False)
_active = set(te.molecule_id.astype(str))
assert _active <= set(map(str, ENG)), 'Engine 2 did not cover all active molecules'
_stage_meta = {}
for _stage, _path in [('ice', 'ice_work/ice_out.json.meta.json'), ('glacier', 'gl_work/gl_out.json.meta.json')]:
    if Path(_path).is_file():
        _stage_meta[_stage] = json.loads(Path(_path).read_text())
    else:
        _stage_meta[_stage] = {'status': 'missing'}
assert _stage_meta['ice'].get('status') == 'ok', _stage_meta['ice']
if (te.adduct == '[M+H]+').any():
    assert _stage_meta['glacier'].get('status') == 'ok', _stage_meta['glacier']

_dm_smiles = _dm_mass = _dm_order = None
_dm_paths = sorted(glob.glob('/kaggle/input/**/dreams_np_mass.npy', recursive=True))
_dreams_used = 0
_audit_rows, _out = [], []
_targets = te.groupby('molecule_id').nm.median().to_dict()
for _row in _before.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = [s.strip() for s in str(_row.smiles).split(';') if s.strip()]
    if _mid not in _active:  # Visible smoke placeholders; hidden rerun processes every molecule.
        _out.append((_row.molecule_id, _row.smiles))
        continue
    _ranks = first_ranks(_original)
    _clean = safe_extend(_original, [])
    _extras = list(FUSION_OVERFLOW.get(_mid, [])) + list(BASE_TAIL_CANDIDATES.get(_mid, []))
    _engine = ENG.get(_mid, {})
    _extras.extend(_engine.get('smiles', []))
    _pc = PC.get(_row.molecule_id, PC.get(_mid, {}))
    _extras.extend(_pc.get('pc', []))
    _revised = safe_extend(_clean, _extras)
    _after_ranked = len(_revised)
    _target = float(_targets.get(_row.molecule_id, _targets.get(_mid, np.nan)))
    if len(_revised) < 25 and np.isfinite(_target) and _target > 0 and _dm_paths:
        if _dm_mass is None:
            _dd = Path(_dm_paths[0]).parent
            _dm_mass = np.load(_dd / 'dreams_np_mass.npy').astype(np.float64)
            _dm_smiles = np.load(_dd / 'dreams_np_smiles.npy', allow_pickle=True)
            assert len(_dm_mass) == len(_dm_smiles)
            _dm_order = np.argsort(_dm_mass, kind='stable')
            _dm_sorted = _dm_mass[_dm_order]
        _lo, _hi = np.searchsorted(_dm_sorted, [_target * (1 - 10e-6), _target * (1 + 10e-6)])
        _ix = _dm_order[_lo:_hi]
        _ix = _ix[np.argsort(np.abs(_dm_mass[_ix] - _target), kind='stable')]
        _mass_candidates = [s.decode() if isinstance(s, bytes) else str(s) for s in _dm_smiles[_ix]]
        _revised = safe_extend(_revised, _mass_candidates)
        _dreams_used += len(_revised) - _after_ranked
    assert _revised, f'No valid candidates for {_mid}'
    assert len(_revised) <= 25 and len(first_ranks(_revised)) == len(_revised)
    assert_monotone(_original, _revised)
    _new_ranks = first_ranks(_revised)
    _audit_rows.append(dict(molecule_id=_mid, original_count=len(_original),
        original_unique=len(_ranks), invalid_count=sum(metric_identity(s) is None for s in _original),
        duplicate_count=sum(metric_identity(s) is not None for s in _original) - len(_ranks),
        revised_count=len(_revised), ranked_added=_after_ranked - len(_clean),
        dreams_added=len(_revised) - _after_ranked,
        promoted_original=sum(_new_ranks[k] < v for k, v in _ranks.items()),
        changed=_original != _revised))
    _out.append((_row.molecule_id, ';'.join(_revised)))
_final = pd.DataFrame(_out, columns=['molecule_id', 'smiles'])
_expected = pd.read_csv(os.path.join(COMP, 'sample_submission.csv')).molecule_id
assert _final.molecule_id.is_unique and set(_final.molecule_id) == set(_expected)
assert _final.smiles.notna().all() and _final.smiles.str.split(';').map(len).between(1, 25).all()
if IS_RERUN:
    assert len(_active) == len(_final), 'A hidden molecule was left unprocessed'
_final.to_csv('submission.csv', index=False)
_audit_df = pd.DataFrame(_audit_rows)
_audit_df.to_csv('candidate_audit.csv', index=False)
_summary = dict(experiment='V1.1 metric-safe tail', rdkit=rdkit.__version__,
    hidden_rerun=IS_RERUN, active_molecules=len(_active), output_rows=len(_final),
    changed_molecules=int(_audit_df.changed.sum()),
    invalid_removed=int(_audit_df.invalid_count.sum()),
    equivalent_duplicates_removed=int(_audit_df.duplicate_count.sum()),
    existing_candidates_promoted=int(_audit_df.promoted_original.sum()),
    ranked_candidates_added=int(_audit_df.ranked_added.sum()),
    dreams_candidates_added=int(_audit_df.dreams_added.sum()),
    monotone_rank_assertions_passed=True,
    metric_source_sha256=METRIC_SOURCE_SHA256,
    baseline_sha256=hashlib.sha256(Path('baseline_submission.csv').read_bytes()).hexdigest(),
    submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
    stages=_stage_meta, runtime_seconds=round(time.time()-T0, 1),
    validation_note='Visible test is a smoke check only and overlaps training; no generalization score claimed.')
Path('candidate_audit.json').write_text(json.dumps(_summary, indent=2, default=str))
print(json.dumps({k:v for k,v in _summary.items() if k != 'stages'}, indent=2))
display(_audit_df)
