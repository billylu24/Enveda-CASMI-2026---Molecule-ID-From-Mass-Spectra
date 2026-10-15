"""Publish the exact reproduction and submit its verified version after user authorization."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent
EVID = ROOT / 'evidence'
STATE = EVID / 'publication_state.json'
KERNEL = 'giaok246/casmi26-flexonafft-evidence-gap-reproduction'
COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'

def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['push', 'status', 'submit', 'submissions']); args = p.parse_args()
    api = KaggleApi(); api.authenticate()
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(kernel=KERNEL)
    def save(): STATE.write_text(json.dumps(state, indent=2, default=str))
    if args.action == 'push':
        assert not state.get('push_intent'), 'Push intent exists; reconcile status before any retry'
        identity = json.loads((EVID / 'release_identity.json').read_text())
        for name, digest in identity['release_files'].items():
            assert hashlib.sha256((ROOT / 'release' / name).read_bytes()).hexdigest() == digest
        state['push_intent'] = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), identity=identity)
        save()
        try:
            response = api.kernels_push(str(ROOT / 'release'), timeout='32400', acc='NvidiaTeslaT4')
        except Exception as error:
            state['push_error'] = dict(error=str(error)); save(); raise
        state['push_response'] = json.loads(str(response)); save(); print(response)
        assert state['push_response'].get('ref', '').removeprefix('/code/') == KERNEL
    elif args.action == 'submit':
        assert not state.get('submission_intent'), 'Submission intent exists; reconcile history before retrying'
        version = state['push_response']['versionNumber']
        acceptance = json.loads((EVID / 'platform_acceptance.json').read_text())
        assert acceptance['execution_acceptance_passed'] and acceptance['version'] == version
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        with api.build_kaggle_client() as k:
            req = ApiGetKernelRequest(); req.user_name, req.kernel_slug = KERNEL.split('/')
            remote = k.kernels.kernels_api_client.get_kernel(req)
        assert remote.metadata.current_version_number == version
        code = lambda nb: [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        assert code(json.loads(remote.blob.source)) == code(json.loads((ROOT / 'release/reproduction.ipynb').read_text()))
        assert json.loads(str(api.kernels_status(KERNEL)))['status'] == 'COMPLETE'
        before = [json.loads(str(x)) for x in api.competition_submissions(COMPETITION, page_size=100)]
        (EVID / 'submissions_before.json').write_text(json.dumps(before, indent=2))
        state['submission_intent'] = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            version=version, user_authorized=True, competition=COMPETITION,
            type='code_submission', output_file='submission.csv',
            acceptance='platform_acceptance.json', upstream_preview_hash_exact=False)
        save()
        try:
            response = api.competition_submit_code(file_name='submission.csv',
                message='Flexonafft V15 Iteration07 source-exact reproduction; pinned original T4 container and 15 inputs; 12-molecule execution verified, one lower-rank preview difference documented; full hidden code rerun.',
                competition=COMPETITION, kernel=KERNEL, kernel_version=version)
        except Exception as error:
            state['submission_error'] = dict(type=type(error).__name__); save(); raise
        state['submission_response'] = json.loads(str(response)); save(); print(response)
    elif args.action == 'submissions':
        records = [json.loads(str(x)) for x in api.competition_submissions(COMPETITION, page_size=100)]
        (EVID / 'competition_submissions.json').write_text(json.dumps(records, indent=2))
        for row in records[:5]: print(json.dumps(row))
    else:
        response = api.kernels_status(KERNEL)
        state['kernel_status'] = json.loads(str(response)); save(); print(response)

if __name__ == '__main__': main()
