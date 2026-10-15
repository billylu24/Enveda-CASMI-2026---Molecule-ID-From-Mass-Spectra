"""Run the private reproduction on Kaggle. No competition-submission operation."""
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

def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['push', 'status']); args = p.parse_args()
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
    else:
        response = api.kernels_status(KERNEL)
        state['kernel_status'] = json.loads(str(response)); save(); print(response)

if __name__ == '__main__': main()
