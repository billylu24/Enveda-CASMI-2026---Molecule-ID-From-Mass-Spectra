"""Observe the frozen source output and preserve same-run fusion inputs for later work."""


def _e004_postflight(scope):
    import hashlib
    import json
    import time
    from pathlib import Path
    import pandas as pd
    import rdkit

    pre = scope['E004_PREFLIGHT']
    ids = set(pre['input_ids'])
    processed = set(scope['te'].molecule_id.astype(str))
    full = bool(scope['IS_RERUN'])
    assert processed == (ids if full else set(sorted(ids)[:scope['SMOKE_N']]))
    frame = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
    assert list(frame.columns) == ['molecule_id', 'smiles']
    assert frame.molecule_id.is_unique and set(frame.molecule_id) == ids
    assert frame.smiles.map(lambda s: 1 <= len(s.split(';')) <= 25).all()
    base, eng, pc = scope['BASE'], scope['ENG'], scope['PC']
    assert set(map(str, base)) == processed
    assert processed <= set(map(str, eng))
    assert scope['n_err'] == 0
    if not full:
        assert all(v[0] for v in base.values())
        assert all(eng[mid]['smiles'] and pc.get(mid, {}).get('pc') for mid in processed)
    stages = {}
    for stage in ('ice', 'gl'):
        path = Path(f'{stage}_work/{stage}_out.json.meta.json')
        meta = json.loads(path.read_text()) if path.is_file() else {'status': 'missing'}
        # The frozen source owns full-run failure policy. Extra acceptance checks
        # only gate the visible smoke, where all expected stages are exercised.
        if not full:
            assert meta['status'] == 'ok' and meta['device'].startswith('cuda'), (stage, meta)
            assert meta['rdkit'] == '2025.03.6'
            assert not meta.get('errors') and meta.get('n_pred_fail', 0) == 0
            assert meta.get('n_feat_fail', 0) == 0
            if meta.get('n_mols_covered', 0):
                assert meta['n_mols_scored'] > 0 and meta['n_scored'] > 0
        stages[stage] = meta
    payload = {
        'prefusion': {mid: scope['PREFUSION_LISTS'][mid] for mid in sorted(processed)},
        'engine2': {mid: eng[mid] for mid in sorted(processed)},
        'pubchem': {mid: pc.get(mid, {}) for mid in sorted(processed)},
        'base_tail': {mid: scope['BASE_TAIL_CANDIDATES'][mid] for mid in sorted(processed)},
        'fusion_overflow': {mid: scope['FUSION_OVERFLOW'].get(mid, []) for mid in sorted(processed)},
    }
    Path('e004_candidate_inputs.json').write_text(json.dumps(payload, indent=2, default=str) + '\n')
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'submission.csv', 'baseline_submission.csv', 'submission_reference.csv',
        'candidate_audit.json', 'experiment_audit.json', 'e004_candidate_inputs.json')}
    report = dict(experiment='E004', source_version_id=356740494, source_version=80,
                  source_code_sha256='eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2',
                  **pre, is_full_run=full, processed_ids=sorted(processed),
                  processed_molecules=len(processed), input_molecules=len(ids),
                  output_rows=len(frame), placeholder_fill_rows=len(ids - processed),
                  base_errors=int(scope['n_err']), base_nonempty=sum(bool(v[0]) for v in base.values()),
                  engine2_nonempty=sum(bool(eng[mid]['smiles']) for mid in processed),
                  pubchem_nonempty=sum(bool(pc.get(mid, {}).get('pc')) for mid in processed),
                  main_rdkit=rdkit.__version__, stages=stages, hashes=hashes,
                  elapsed_seconds=round(time.time() - scope['T0'], 2),
                  limitations=['Visible smoke is not independent validation.',
                               'Author 0.436 is not this reproduction score.',
                               'Whole v80 recipe differs from E003 in multiple factors.'])
    Path('e004_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('E004_VERIFICATION', json.dumps(report))
    return report


E004_REPORT = _e004_postflight(globals())
del _e004_postflight
