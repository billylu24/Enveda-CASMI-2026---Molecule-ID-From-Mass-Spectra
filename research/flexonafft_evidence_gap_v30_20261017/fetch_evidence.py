"""Download selected small public/runtime evidence; never persist signed URLs."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
from prepare_release import ROOT,SOURCE,KERNEL,VERSION,code

NAMES={'submission.csv','submission_v22_control.csv','submission_base_e2r2.csv','eng/bio_union_audit.json',
       'v22_adaptation_audit.json','e2r2_stats.json','e004_verification.json','experiment_audit_factorial.json',
       'experiment_verification.json','experiment_audit.json','e004_candidate_inputs.json','v22_comparison.csv',
       'reproduction_acceptance.json','processed_predictions.csv',
       'ice_work/ice_out.json.meta.json','gl_work/gl_out.json.meta.json'}

def fetch(mode):
    api=KaggleApi();api.authenticate()
    ref,version=(SOURCE,VERSION) if mode=='source' else (KERNEL,json.loads((ROOT/'evidence/publication_state.json').read_text())['push_response']['versionNumber'])
    dest=ROOT/'evidence'/('source_reference' if mode=='source' else 'platform');dest.mkdir(exist_ok=True)
    def metadata():
        with api.build_kaggle_client() as k:
            r=ApiGetKernelRequest();r.user_name,r.kernel_slug=ref.split('/')
            response=k.kernels.kernels_api_client.get_kernel(r)
        meta=json.loads(str(response.metadata));assert meta['currentVersionNumber']==version,'Remote version changed; refuse mixed evidence'
        (dest/'api_metadata.json').write_text(json.dumps(meta,indent=2))
        (dest/'platform_source.ipynb').write_text(response.blob.source)
        return response
    metadata();status=json.loads(str(api.kernels_status(ref)));(dest/'kernel_status.json').write_text(json.dumps(status,indent=2))
    token=None;manifest=[];selected=[]
    with api.build_kaggle_client() as k:
        while True:
            req=ApiListKernelSessionOutputRequest();req.user_name,req.kernel_slug=ref.split('/');req.page_size=100;req.page_token=token
            response=k.kernels.kernels_api_client.list_kernel_session_output(req)
            if response.log:
                (dest/'runtime_log.json').write_text(response.log)
            for item in response.files or []:
                manifest.append(item.file_name)
                if item.file_name in NAMES:selected.append((item.file_name,item.url))
            token=response.next_page_token
            if not token:break
    metadata()
    (dest/'output_manifest.json').write_text(json.dumps(dict(files=manifest,selected=[n for n,_ in selected]),indent=2))
    def download(item):
        name,url=item;path=dest/name;assert path.resolve().is_relative_to(dest.resolve())
        try:
            r=requests.get(url,timeout=(8,25));r.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
            return dict(name=name,downloaded=True,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
        except requests.RequestException as error:return dict(name=name,downloaded=False,error_type=type(error).__name__)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(download,selected))
    (dest/'evidence_transport.json').write_text(json.dumps(dict(version=version,status=status,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=records),indent=2))
    print(json.dumps(dict(ref=ref,status=status,files=records),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['source','run']);fetch(parser.parse_args().mode)
