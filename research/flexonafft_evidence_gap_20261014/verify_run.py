"""Verify exact code, completed GPU execution and cross-run prediction identity.

When output-file hosting is unavailable, compare hashes computed and printed by
both runtime processes, and explicitly record that no local CSV-byte comparison
was possible. Never equate placeholder rows with processed molecules.
"""
import datetime
import hashlib
import json
from pathlib import Path
import pandas as pd
from fetch_evidence import ROOT, EVID


def read(path): return json.loads(path.read_text())
def code(nb): return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    platform = EVID / 'platform'; source = EVID / 'source_reference'
    state = read(EVID / 'publication_state.json')
    assert state['kernel_status']['status'] == 'COMPLETE', 'GPU execution has not completed'
    version = state['push_response']['versionNumber']
    assert read(platform / 'api_metadata.json')['currentVersionNumber'] == version
    assert code(read(platform / 'platform_source.ipynb')) == code(read(ROOT / 'release/reproduction.ipynb'))
    assert code(read(source / 'platform_source.ipynb')) == code(read(ROOT / 'source/casmi26-closing-the-evidence-gap.ipynb'))
    assert read(source / 'api_metadata.json')['currentVersionNumber'] == 15
    assert code(read(ROOT / 'release/reproduction.ipynb'))[:-1] == code(read(ROOT / 'source/casmi26-closing-the-evidence-gap.ipynb'))
    accept = read(platform / 'reproduction_acceptance.json')
    assert accept['passed'] and accept['source_version'] == 15 and accept['original_code_cells'] == 9
    assert accept['rows'] == 400 and accept['processed'] == 12 and accept['placeholders'] == 388
    assert not accept['hidden_rerun'] and accept['base_errors'] == 0
    assert accept['main_rdkit'] == '2026.03.3'
    for meta in accept['simulators'].values():
        assert meta['status'] == 'ok' and meta['device'].startswith('cuda')
        assert meta['rdkit'] == '2025.03.6' and not meta.get('errors')
        assert meta.get('n_pred_fail', 0) == meta.get('n_feat_fail', 0) == 0
        if meta.get('n_mols_covered', 0):
            assert meta['n_mols_scored'] > 0 and meta['n_scored'] > 0
    original = read(source / 'iteration_audit.json')
    actual = read(platform / 'iteration_audit.json')
    assert actual == accept['iteration07']
    hash_exact = actual['submission_sha256'] == original['submission_sha256']
    for key in ['experiment', 'changed_rows', 'changed_top1', 'details']:
        assert actual[key] == original[key], (key, actual[key], original[key])
    assert accept['initial01']['source_control_reconstructed']
    assert accept['initial01']['routes'] == read(source / 'research_routing_audit.json')['routes']
    files = {}
    local = platform / 'submission.csv'; upstream = source / 'submission.csv'
    for path in [local, upstream]:
        if not path.exists(): continue
        expected = actual if path == local else original
        assert sha(path) == expected['submission_sha256']
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(df.columns) == ['molecule_id', 'smiles']
        assert len(df) == 400 and df.molecule_id.is_unique
        assert df.smiles.str.split(';').map(len).between(1, 25).all()
        files[str(path.relative_to(ROOT))] = sha(path)
    comparison = None
    if local.exists() and upstream.exists():
        a = pd.read_csv(local, dtype=str).set_index('molecule_id')
        b = pd.read_csv(upstream, dtype=str).set_index('molecule_id')
        assert list(a.index) == list(b.index)
        active_ids = read(platform / 'e004_verification.json')['processed_ids']
        a.loc[active_ids].reset_index().to_csv(platform / 'processed_predictions.csv', index=False)
        b.loc[active_ids].reset_index().to_csv(source / 'processed_predictions.csv', index=False)
        changed = []
        for mid in active_ids:
            first, second = a.loc[mid, 'smiles'].split(';'), b.loc[mid, 'smiles'].split(';')
            if first != second:
                changed.append(dict(molecule_id=mid, top1_equal=first[:1] == second[:1],
                    candidate_set_equal=set(first) == set(second),
                    differing_ranks=[i+1 for i in range(max(len(first), len(second)))
                                     if first[i:i+1] != second[i:i+1]],
                    source_candidates=second, reproduction_candidates=first))
        actual_inputs = read(platform / 'e004_candidate_inputs.json')
        source_inputs = read(source / 'e004_candidate_inputs.json')
        channel_differences = {channel: [mid for mid in source_inputs[channel]
            if source_inputs[channel][mid] != actual_inputs[channel].get(mid)] for channel in source_inputs}
        comparison = dict(processed_molecules=len(active_ids), exact_rank_matches=len(active_ids)-len(changed),
            changed_rows=len(changed), top1_matches=len(active_ids)-sum(not x['top1_equal'] for x in changed),
            candidate_set_matches=len(active_ids)-sum(not x['candidate_set_equal'] for x in changed),
            placeholder_rows_excluded=388, changed_details=changed,
            source_submission_sha256=original['submission_sha256'],
            reproduction_submission_sha256=actual['submission_sha256'],
            channel_differences=channel_differences, output_byte_identical=hash_exact,
            cause='Difference localized to main prefusion/base-tail candidate order; exact numerical mechanism not established. No algorithm changed.')
        (EVID / 'source_preview_comparison.json').write_text(json.dumps(comparison, indent=2))
    report = dict(passed=True, execution_acceptance_passed=True, version=version, source_version=15, original_code_exact=True,
        completed_gpu_run=True, processed_molecules=12, placeholder_rows_excluded=388,
        runtime_reported_whole_csv_hash_exact=hash_exact,
        full_output_parity_passed=hash_exact, source_preview_comparison=comparison,
        downloaded_csv_bytes_verified=local.exists() and upstream.exists(),
        submission_sha256=actual['submission_sha256'], changed_rows=actual['changed_rows'],
        changed_top1=actual['changed_top1'], target_groups=actual['details']['target_groups'],
        accepted_groups=actual['details']['accepted_groups'],
        local_files_sha256=files, measured_score=None,
        artifact_transport=read(platform / 'evidence_transport.json'),
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (EVID / 'platform_acceptance.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
