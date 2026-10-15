"""Fetch versioned public provenance and selected small outputs; never log signed URLs."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest

ROOT = Path(__file__).resolve().parent
EVID = ROOT / 'evidence'
SOURCE = 'flexonafft/casmi26-closing-the-evidence-gap'
KERNEL = 'giaok246/casmi26-flexonafft-evidence-gap-reproduction'
NAMES = {'e004_candidate_inputs.json', 'baseline_submission.csv', 'submission.csv', 'submission_0441_control.csv', 'submission_v7c_control.csv',
         'iteration_audit.json', 'iteration_vs_0441.csv', 'iteration_group_audit.csv',
         'research_routing_audit.json', 'research_routing_audit.csv', 'research_evidence_report.png',
         'e004_verification.json', 'reproduction_acceptance.json',
         'ice_work/ice_out.json.meta.json', 'gl_work/gl_out.json.meta.json'}

def client():
    api = KaggleApi(); api.authenticate(); return api

def read_source(api, ref, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with api.build_kaggle_client() as k:
        req = ApiGetKernelRequest(); req.user_name, req.kernel_slug = ref.split('/')
        result = k.kernels.kernels_api_client.get_kernel(req)
    meta = json.loads(str(result.metadata))
    (dest / 'api_metadata.json').write_text(json.dumps(meta, indent=2))
    (dest / 'platform_source.ipynb').write_text(result.blob.source)
    return meta

def freeze_dependencies():
    api = client()
    meta = json.loads((EVID / 'source_api_metadata.json').read_text())
    cutoff = meta['lastRunTime']
    def get(ref):
        with api.build_kaggle_client() as k:
            req = ApiGetDatasetRequest(); req.owner_slug, req.dataset_slug = ref.split('/')
            data = json.loads(str(k.datasets.dataset_api_client.get_dataset(req)))
        versions = [v for v in data['versions'] if v['creationDate'] <= cutoff and v['status'] == 'Ready']
        version = max(versions, key=lambda v: v['versionNumber'])
        return dict(ref=ref, version=version['versionNumber'], created=version['creationDate'],
                    license=data['licenseName'], dataset_id=data['id'],
                    selection='Latest ready data version created before source lastRunTime; original attachment version unavailable from public API')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(get, meta['datasetDataSources']))
    (EVID / 'dependencies.json').write_text(json.dumps(items, indent=2))
    print(json.dumps(items, indent=2))

def extract_logged_reports(log, dest):
    """Preserve reports printed by the run; record when artifact transport is unavailable."""
    rows = json.loads(log or '[]')
    text = ''.join(row.get('data', '') for row in rows)
    decoder = json.JSONDecoder(); reports = {}
    for i, char in enumerate(text):
        if char != '{': continue
        try: data, _ = decoder.raw_decode(text[i:])
        except ValueError: continue
        if not isinstance(data, dict): continue
        if data.get('experiment') == 'replicated_fragment_fallback' and 'details' in data:
            reports['iteration_audit.json'] = data
        if data.get('experiment') == 'initial-faithful' and 'source_control_reconstructed' in data:
            reports['research_routing_audit.json'] = data
        if data.get('experiment') == 'E004' and 'processed_molecules' in data:
            reports['e004_verification.json'] = data
        if data.get('passed') and data.get('source') == SOURCE and 'iteration07' in data:
            reports['reproduction_acceptance.json'] = data
    for name, data in reports.items():
        (dest / name).write_text(json.dumps(data, indent=2))
    return sorted(reports)

def outputs(api, ref, version, dest):
    meta = read_source(api, ref, dest)
    assert meta['currentVersionNumber'] == version, 'Version changed; do not compare mixed versions'
    token = None; manifest = []; selected = []; log = None
    with api.build_kaggle_client() as k:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = ref.split('/'); req.page_size = 100; req.page_token = token
            result = k.kernels.kernels_api_client.list_kernel_session_output(req)
            if result.log:
                log = result.log
                (dest / 'pipeline.log').write_text(log)
                (dest / 'runtime_log.json').write_text(log)
                logs = json.loads(log)
                (dest / 'pipeline_readable.log').write_text(''.join(f"[{x.get('time', 0):.1f}s] {x.get('data', '')}" for x in logs))
            for item in result.files or []:
                manifest.append(dict(name=item.file_name))
                if item.file_name in NAMES: selected.append((item.file_name, item.url))
            token = result.next_page_token
            if not token: break
    assert read_source(api, ref, dest)['currentVersionNumber'] == version, 'Version changed during output retrieval'
    (dest / 'output_manifest.json').write_text(json.dumps(dict(listed_file_count=len(manifest), selected_files=[x for x in manifest if x['name'] in NAMES]), indent=2))
    logged = extract_logged_reports(log, dest) if log else []
    def download(item):
        name, url = item
        path = dest / name
        assert path.resolve().is_relative_to(dest.resolve())
        try:
            response = requests.get(url, timeout=(8, 12)); response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(response.content)
            return dict(name=name, origin='download', bytes=len(response.content),
                        sha256=hashlib.sha256(response.content).hexdigest())
        except requests.RequestException as error:
            # Never include error strings: they can contain credential-bearing output URLs.
            return dict(name=name, origin='runtime_log' if name in logged else 'unavailable',
                        transport_error=type(error).__name__)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, selected))
    (dest / 'evidence_transport.json').write_text(json.dumps(dict(version=version,
        logged_reports=logged, files=records), indent=2))
    for record in records: print(record['origin'].upper(), record['name'], flush=True)
    return {r['name'] for r in records if r['origin'] == 'download'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['dependencies', 'source', 'run']); a = p.parse_args()
    EVID.mkdir(exist_ok=True)
    if a.action == 'dependencies': freeze_dependencies()
    else:
        api = client(); ref = SOURCE if a.action == 'source' else KERNEL
        version = 15 if a.action == 'source' else json.loads((EVID / 'publication_state.json').read_text())['push_response']['versionNumber']
        outputs(api, ref, version, EVID / ('source_reference' if a.action == 'source' else 'platform'))
