"""Upload a private Kaggle notebook once; read-only status on subsequent calls."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from prepare_release import ROOT, KERNEL, code

STATE = ROOT/'evidence/publication_state.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['push','status']);args=p.parse_args()
    api=KaggleApi();api.authenticate()
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(kernel=KERNEL)
    def save(): STATE.write_text(json.dumps(state,indent=2))
    if args.action=='push':
        assert not state.get('push_intent'), 'Prior push intent exists: reconcile remote status before retrying'
        identity=json.loads((ROOT/'evidence/release_identity.json').read_text())
        for name,digest in identity['release_files'].items():
            assert hashlib.sha256((ROOT/'release'/name).read_bytes()).hexdigest()==digest
        state['push_intent']=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),user_authorized=True,identity=identity)
        save()
        try:
            response=api.kernels_push(str(ROOT/'release'),timeout='32400',acc='NvidiaTeslaT4')
        except Exception as error:
            state['push_error_type']=type(error).__name__;save();raise
        state['push_response']=json.loads(str(response));save();print(response)
        assert state['push_response']['ref'].removeprefix('/code/')==KERNEL
    else:
        state['kernel_status']=json.loads(str(api.kernels_status(KERNEL)))
        with api.build_kaggle_client() as k:
            r=ApiGetKernelRequest();r.user_name,r.kernel_slug=KERNEL.split('/')
            remote=k.kernels.kernels_api_client.get_kernel(r)
        meta=json.loads(str(remote.metadata));state['remote_version']=meta['currentVersionNumber']
        assert code(json.loads(remote.blob.source))==code(json.loads((ROOT/'release/reproduction.ipynb').read_text()))
        state['remote_code_exact']=True;state['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps({k:v for k,v in state.items() if k not in ['push_intent']},indent=2))

if __name__=='__main__': main()
