"""Continue the explicitly requested full reproduction through one code submission and score."""
import datetime, hashlib, json, re, subprocess, sys, time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest, ApiGetKernelRequest
ROOT=Path(__file__).resolve().parent;EVID=ROOT/'evidence';OUT=EVID/'platform_output'
KERNEL='giaok246/casmi26-samanyu-iceberg-0-438-reproduction';COMP='enveda-CASMI26-molecule-id-mass-spectra'
api=KaggleApi();api.authenticate()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def retry(fn):
 for attempt in range(10):
  try:return fn()
  except (requests.RequestException,ConnectionError,OSError) as e:
   if isinstance(e,requests.HTTPError) and e.response.status_code not in {408,429,500,502,503,504}:raise
   print(stamp(),'TRANSIENT',type(e).__name__,attempt,flush=True)
   if attempt==9:raise
   time.sleep(min(5*(attempt+1),30))
def get_outputs(version):
 OUT.mkdir(exist_ok=True)
 owner,slug=KERNEL.split('/')
 names={'submission.csv','run_manifest.json','reproduction_acceptance.json','eng_lists.json','pc_lists.json','baseline_submission.csv','candidate_audit.csv','candidate_audit.json','ice_work/ice_out.json.meta.json','gl_work/gl_out.json.meta.json'}
 downloaded=set()
 with api.build_kaggle_client() as k:
  q=ApiGetKernelRequest();q.user_name=owner;q.kernel_slug=slug
  r=retry(lambda:k.kernels.kernels_api_client.get_kernel(q))
  assert r.metadata.current_version_number==version, 'Latest version changed'
  (OUT/'api_metadata.json').write_text(str(r.metadata))
  (OUT/'platform_source.ipynb').write_text(r.blob.source)
  local=json.loads((ROOT/'release/reproduction.ipynb').read_text());remote=json.loads(r.blob.source)
  getcode=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
  assert getcode(local)==getcode(remote), 'Remote source mismatch'
  token=None
  while True:
   q=ApiListKernelSessionOutputRequest();q.user_name=owner;q.kernel_slug=slug;q.page_size=100;q.page_token=token
   r=retry(lambda:k.kernels.kernels_api_client.list_kernel_session_output(q))
   if r.log:
    (OUT/'pipeline.log').write_text(r.log)
    try:
     items=json.loads(r.log)
     (OUT/'pipeline_readable.log').write_text(''.join(f"[{x.get('time',0):.1f}s] {x.get('data','')}" for x in items))
    except (ValueError,TypeError):pass
   for item in r.files or []:
    if item.file_name not in names:continue
    dest=OUT/item.file_name;assert dest.resolve().is_relative_to(OUT.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
    def fetch():
     r=requests.get(item.url,timeout=(10,30));r.raise_for_status();return r.content
    dest.write_bytes(retry(fetch));downloaded.add(item.file_name);print(stamp(),'DOWNLOADED',item.file_name,flush=True)
   token=r.next_page_token
   if not token:break
   time.sleep(2)
 assert names<=downloaded, ('Missing acceptance outputs',names-downloaded)
 acceptance=json.loads((OUT/'reproduction_acceptance.json').read_text())
 assert acceptance['passed'] and acceptance['processed']==12 and acceptance['rows']==400 and acceptance['preview_placeholders']==388 and not acceptance['hidden_rerun']
 assert acceptance['base_errors']==0 and acceptance['engine2_molecules']==acceptance['pubchem_molecules']==12
 assert acceptance['rdkit']=='2026.03.3' and acceptance['candidate_max']<=25
 for m in acceptance['stages'].values():assert m['status']=='ok' and m['scored']==m['covered'] and m['scored']>0
 assert hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()==acceptance['submission_sha256']
 import pandas as pd
 df=pd.read_csv(OUT/'submission.csv');sample=pd.read_csv(ROOT.parents[1]/'data/sample_submission.csv')
 assert df.molecule_id.is_unique and list(df.molecule_id)==list(sample.molecule_id)
 assert df.smiles.notna().all() and df.smiles.str.split(';').map(len).between(1,25).all()
 reference=pd.read_csv(EVID/'source_reference/submission.csv').set_index('molecule_id')
 active_ids=pd.read_csv(EVID/'source_reference/candidate_audit.csv').molecule_id.tolist()
 actual=df.set_index('molecule_id')
 equal=[mid for mid in active_ids if actual.loc[mid,'smiles']==reference.loc[mid,'smiles']]
 comparison=dict(processed_reference_molecules=len(active_ids), exact_rank_matches=len(equal), different_ids=[mid for mid in active_ids if mid not in equal], placeholder_rows_excluded=388, upstream_preview_sha256=hashlib.sha256((EVID/'source_reference/submission.csv').read_bytes()).hexdigest(), reproduction_preview_sha256=hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest())
 comparison['whole_preview_csv_exact']=comparison['upstream_preview_sha256']==comparison['reproduction_preview_sha256']
 assert comparison['whole_preview_csv_exact']
 (EVID/'source_preview_comparison.json').write_text(json.dumps(comparison,indent=2))
 assert len(equal)==len(active_ids)==12, ('Source public preview differs',comparison)
 evidence=dict(passed=True,version=version,remote_code_exact=True,visible_preview_molecules=12,hidden_completeness_assertions=True,acceptance=acceptance,files_sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in downloaded},checked_at=stamp())
 (EVID/'platform_acceptance.json').write_text(json.dumps(evidence,indent=2))
 print(stamp(),'SOURCE_EXACT_PREVIEW_ACCEPTANCE_PASSED',json.dumps(acceptance),flush=True)
s=json.loads((EVID/'publication_state.json').read_text());version=s['push_response']['versionNumber']
while True:
 status=json.loads(str(retry(lambda:api.kernels_status(KERNEL))))
 (EVID/'monitor_status.json').write_text(json.dumps(dict(time=stamp(),phase='notebook',status=status,version=version),indent=2))
 print(stamp(),'KERNEL',status,flush=True)
 if status['status']=='COMPLETE':break
 if status['status'] not in ['RUNNING','QUEUED']:raise RuntimeError(status)
 time.sleep(45)
get_outputs(version)
s=json.loads((EVID/'publication_state.json').read_text())
if not s.get('submission_intent'):
 subprocess.run([sys.executable,str(ROOT/'publish.py'),'submit','--version',str(version)],check=True)
s=json.loads((EVID/'publication_state.json').read_text());response=s.get('submission_response',{})
print(stamp(),'SUBMISSION_RESPONSE',response,flush=True)
submission_id=response.get('ref') or response.get('submissionId') or response.get('id')
while True:
 rows=retry(lambda:api.competition_submissions(COMP,page_size=100));records=[json.loads(str(x)) for x in rows]
 (EVID/'competition_submissions.json').write_text(json.dumps(records,indent=2))
 matches=[x for x in records if str(x.get('ref'))==str(submission_id)] if submission_id else []
 if not matches:
  matches=[x for x in records if x.get('submittedFileName')=='submission.csv' and 'Samanyu ICEBERG where GLACIER is blind public V1 reproduction' in str(x.get('description',''))]
 assert len(matches)<=1, 'Ambiguous matching submission; reconcile manually'
 if matches:
  record=matches[0];(EVID/'submission_result.json').write_text(json.dumps(record,indent=2))
  print(stamp(),'SUBMISSION',record,flush=True)
  (EVID/'monitor_status.json').write_text(json.dumps(dict(time=stamp(),phase='submission',submission=record),indent=2))
  if str(record.get('status','')).upper() in ['COMPLETE','ERROR','CANCELLED']:
   break
 else:print(stamp(),'WAITING_FOR_SUBMISSION_RECORD',submission_id,flush=True)
 time.sleep(45)
if str(record.get('status','')).upper()=='COMPLETE' and record.get('publicScore') is not None:
 subprocess.run([sys.executable,str(ROOT/'finalize_report.py')],check=True)
 subprocess.run([sys.executable,str(ROOT/'finish_git.py')],check=True)
print(stamp(),'FINISHED',flush=True)
