"""Read exact-version Kaggle source and bounded output evidence without signed URLs."""
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest

ROOT = Path(__file__).resolve().parent
EVID = ROOT / 'evidence'
SOURCE = 'canqiang/casmi26-e006-window-40-ice-fallback-true'
KERNEL = 'giaok246/casmi26-canqiang-e006-reproduction'
NAMES = {'submission.csv', 'baseline_submission.csv', 'submission_reference.csv', 'submission_e004_control.csv',
         'candidate_audit.csv', 'candidate_audit.json', 'run_manifest.json', 'experiment_audit.csv',
         'experiment_audit.json', 'experiment_audit_factorial.csv', 'experiment_audit_factorial.json',
         'experiment_verification.json', 'e004_verification.json', 'reproduction_acceptance.json',
         'ice_work/ice_out.json.meta.json', 'gl_work/gl_out.json.meta.json'}

def api():
    client = KaggleApi()
    client.authenticate()
    return client

def read_source(client, kernel, version, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with client.build_kaggle_client() as k:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = kernel.split('/')
        result = k.kernels.kernels_api_client.get_kernel(request)
    assert result.metadata.current_version_number == version, 'Remote version changed'
    (dest / 'api_metadata.json').write_text(str(result.metadata))
    (dest / 'platform_source.ipynb').write_text(result.blob.source)
    return result

def download_outputs(client, kernel, version, dest, required):
    dest.mkdir(parents=True, exist_ok=True)
    read_source(client, kernel, version, dest)
    token = None
    found = set()
    manifest = []
    with client.build_kaggle_client() as k:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = kernel.split('/')
            request.page_size = 100
            request.page_token = token
            result = k.kernels.kernels_api_client.list_kernel_session_output(request)
            if result.log:
                (dest / 'pipeline.log').write_text(result.log)
                try:
                    items = json.loads(result.log)
                    (dest / 'pipeline_readable.log').write_text(''.join(f"[{x.get('time',0):.1f}s] {x.get('data','')}" for x in items))
                except (ValueError, TypeError):
                    pass
            for item in result.files or []:
                manifest.append(dict(name=item.file_name))
                if item.file_name not in NAMES:
                    continue
                path = dest / item.file_name
                assert path.resolve().is_relative_to(dest.resolve())
                path.parent.mkdir(parents=True, exist_ok=True)
                response = requests.get(item.url, timeout=(10, 45))
                response.raise_for_status()
                path.write_bytes(response.content)
                found.add(item.file_name)
                print('DOWNLOADED', item.file_name, flush=True)
            token = result.next_page_token
            print('OUTPUT_PAGE', len(manifest), 'more', bool(token), flush=True)
            if not token:
                break
    (dest / 'output_manifest.json').write_text(json.dumps(manifest, indent=2))
    assert set(required) <= found, ('Missing evidence', set(required) - found)
    return found

if __name__ == '__main__':
    client = api()
    download_outputs(client, SOURCE, 1, EVID / 'source_reference', NAMES - {'reproduction_acceptance.json'})
