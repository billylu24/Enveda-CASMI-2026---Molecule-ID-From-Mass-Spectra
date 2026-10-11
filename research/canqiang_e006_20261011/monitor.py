"""Finish one source-exact Kaggle reproduction, code submission and official score."""
import datetime
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import requests
import pandas as pd
from kaggle_evidence import api, download_outputs, read_source, KERNEL, EVID, ROOT, NAMES

COMP = 'enveda-CASMI26-molecule-id-mass-spectra'
STATE = EVID / 'publication_state.json'
OUT = EVID / 'platform_output'

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str))

def retry(fn):
    for attempt in range(6):
        try:
            return fn()
        except (requests.RequestException, ConnectionError, OSError) as error:
            if isinstance(error, requests.HTTPError) and error.response.status_code not in {408, 429, 500, 502, 503, 504}:
                raise
            print(stamp(), 'TRANSIENT', type(error).__name__, attempt, flush=True)
            if attempt == 5:
                raise
            time.sleep(min(5 * (attempt + 1), 30))

def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

def accept(client, version):
    retry(lambda: download_outputs(client, KERNEL, version, OUT, NAMES))
    local = json.loads((ROOT / 'release/reproduction.ipynb').read_text())
    remote = json.loads((OUT / 'platform_source.ipynb').read_text())
    assert code(local) == code(remote), 'Remote code differs'
    acceptance = json.loads((OUT / 'reproduction_acceptance.json').read_text())
    assert acceptance['passed'] and acceptance['processed'] == 12 and acceptance['rows'] == 400
    assert acceptance['preview_placeholders'] == 388 and not acceptance['hidden_rerun']
    assert acceptance['base_errors'] == 0 and acceptance['main_rdkit'] == '2026.03.3'
    assert acceptance['window'] == 40 and acceptance['ice_fallback']
    for stage in acceptance['stages'].values():
        assert stage['status'] == 'ok' and stage['device'].startswith('cuda')
        assert stage['rdkit'] == '2025.03.6'
        assert not stage.get('errors') and stage.get('n_pred_fail', 0) == stage.get('n_feat_fail', 0) == 0
        assert stage['n_mols_scored'] == stage['n_mols_covered'] > 0
    actual_sha = hashlib.sha256((OUT / 'submission.csv').read_bytes()).hexdigest()
    upstream = json.loads((EVID / 'source_reference/experiment_verification.json').read_text())
    assert actual_sha == acceptance['submission_sha256'] == upstream['submission_sha256'], 'Source preview mismatch'
    reference = pd.read_csv(EVID / 'source_reference/submission.csv', dtype=str)
    actual = pd.read_csv(OUT / 'submission.csv', dtype=str)
    assert actual.equals(reference) and actual.molecule_id.is_unique
    ids = upstream['processed_ids']
    assert ids == acceptance['factorial']['processed_ids']
    assert actual.smiles.str.split(';').map(len).between(1, 25).all()
    comparison = dict(source_version=1, compared_active_molecules=len(ids), exact_active_rank_matches=len(ids),
                      whole_preview_csv_exact=True, preview_placeholders_excluded=388,
                      source_sha256=upstream['submission_sha256'], reproduction_sha256=actual_sha)
    write(EVID / 'source_preview_comparison.json', comparison)
    evidence = dict(passed=True, version=version, remote_code_exact=True, original_code_cells=30,
                    source_preview_exact=True, acceptance=acceptance,
                    files_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in NAMES},
                    checked_at=stamp())
    write(EVID / 'platform_acceptance.json', evidence)
    print(stamp(), 'SOURCE_EXACT_ACCEPTANCE_PASSED', comparison, flush=True)

lock = (EVID / 'monitor.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
client = api()
state = json.loads(STATE.read_text())
version = state['push_response']['versionNumber']
if not (EVID / 'platform_acceptance.json').exists():
    while True:
        status = json.loads(str(retry(lambda: client.kernels_status(KERNEL))))
        write(EVID / 'monitor_status.json', dict(time=stamp(), phase='notebook', status=status, version=version))
        print(stamp(), 'KERNEL', status, flush=True)
        if status['status'] == 'COMPLETE':
            break
        if status['status'] not in {'RUNNING', 'QUEUED'}:
            raise RuntimeError(status)
        time.sleep(45)
    accept(client, version)
state = json.loads(STATE.read_text())
if not state.get('submission_intent'):
    subprocess.run([sys.executable, str(ROOT / 'publish.py'), 'submit', '--version', str(version)], check=True)
state = json.loads(STATE.read_text())
response = state.get('submission_response', {})
submission_id = response.get('ref') or response.get('submissionId') or response.get('id')
assert submission_id, 'Submission intent exists without response: reconcile before retry'
print(stamp(), 'SUBMISSION_RESPONSE', response, flush=True)
pending_synced = (EVID / 'pending_summary.json').exists()
while True:
    rows = retry(lambda: client.competition_submissions(COMP, page_size=100))
    records = [json.loads(str(row)) for row in rows]
    matches = [row for row in records if str(row.get('ref')) == str(submission_id)]
    assert len(matches) <= 1
    if matches:
        record = matches[0]
        write(EVID / 'submission_result.json', record)
        write(EVID / 'monitor_status.json', dict(time=stamp(), phase='submission', submission=record))
        print(stamp(), 'SUBMISSION', {key: record.get(key) for key in ['ref', 'status', 'publicScore', 'errorDescription']}, flush=True)
        if record.get('status') in {'PENDING', 'RUNNING'} and not pending_synced:
            subprocess.run([sys.executable, str(ROOT / 'finalize_report.py'), '--pending'], check=True)
            subprocess.run([sys.executable, str(ROOT / 'finish_git.py'), '--pending'], check=True)
            pending_synced = True
        if str(record.get('status', '')).upper() in {'COMPLETE', 'ERROR', 'CANCELLED'}:
            break
    time.sleep(45)
assert str(record['status']).upper() == 'COMPLETE', record
assert record.get('publicScore') is not None, record
subprocess.run([sys.executable, str(ROOT / 'finalize_report.py')], check=True)
subprocess.run([sys.executable, str(ROOT / 'finish_git.py')], check=True)
print(stamp(), 'FINISHED', flush=True)
