'''Independent experimental overlay
One vote per official metric identity per engine; compact unique ranks before the 40-candidate fusion shortlist. Preserve the existing PubChem gate and same-formula GLACIER logic. This tests duplicate-vote bias, not a coefficient search. It can change top-1 and reduce the score.'''

def metric_fusion(first, second, identity, alpha=0.6, offset=3.0, limit=40):
    scores, representatives = {}, {}
    collapsed = 0
    for candidates, weight in ((first, 1.0), (second, alpha)):
        seen = set()
        for candidate in candidates:
            key = identity(candidate)
            if key is None:
                continue
            if key in seen:
                collapsed += 1
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (offset + len(seen))
            representatives.setdefault(key, candidate)
    ordered = sorted(scores, key=lambda key: -scores[key])[:limit]
    return [representatives[key] for key in ordered], [scores[key] for key in ordered], collapsed

EXPERIMENT = 'metric_identity_fusion'
_reference = pd.read_csv('submission.csv')
_reference.to_csv('submission_reference.csv', index=False)
_rows, _audit = [], []
for _row in _reference.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = str(_row.smiles).split(';')
    _revised, _collapsed = list(_original), 0
    _engine = ENG.get(_mid)
    if _mid in PREFUSION_LISTS and _engine and _engine.get('smiles'):
        _first = [s for s in PREFUSION_LISTS[_mid] if s and s != 'CCO']
        _candidates, _scores, _collapsed = metric_fusion(_first, _engine['smiles'], metric_identity,
                                                       alpha=ALPHA, offset=KRR)
        _keys = [chem.score_key(s) or s for s in _candidates]
        _ice = ICE_SCORES.get(_mid, {}) if isinstance(ICE_SCORES, dict) else {}
        _gl = gl_of(_mid)
        if _ice and _gl:
            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,
                [_form2(s) for s in _candidates], [{}, _gl], [ICE_LAM, GL_LAM], top_n=len(_candidates))
            _candidates = [_candidates[i] for i in _order]
        if _candidates:
            _revised = safe_extend(_candidates[:25], _candidates[25:] + _original)
    _rows.append((_row.molecule_id, ';'.join(_revised)))
    _audit.append(dict(molecule_id=_mid, collapsed_source_votes=_collapsed,
        changed=_revised != _original, changed_top1=_revised[:1] != _original[:1]))

_result = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
assert list(_result.molecule_id) == list(_reference.molecule_id)
assert _result.molecule_id.is_unique and _result.smiles.notna().all()
assert _result.smiles.map(lambda value: 1 <= len(value.split(';')) <= 25).all()
_result.to_csv('submission.csv', index=False)
pd.DataFrame(_audit).to_csv('experiment_audit.csv', index=False)
_changed = sum(row['changed'] for row in _audit)
_report = dict(experiment=EXPERIMENT, changed_molecules=_changed,
               changed_top1=sum(row['changed_top1'] for row in _audit),
               reference_sha256=sha256('submission_reference.csv'),
               submission_sha256=sha256('submission.csv'), measured_score=None)
json.dump(_report, open('experiment_audit.json', 'w'), indent=2)
print(_report)
if not _changed:
    print('NO-OP: predictions match the reference; do not spend a second submission on this file.')
