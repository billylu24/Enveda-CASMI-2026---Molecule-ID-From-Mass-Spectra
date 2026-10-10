"""Publish and submit one exact notebook version, with persisted intent and completion checks."""
import argparse, datetime, hashlib, json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
ROOT=Path(__file__).resolve().parent
EVID=ROOT/'evidence';STATE=EVID/'publication_state.json'
COMP='enveda-CASMI26-molecule-id-mass-spectra'
KERNEL='giaok246/casmi26-samanyu-iceberg-0-438-reproduction'
p=argparse.ArgumentParser();p.add_argument('action',choices=['push','status','submit','submissions']);p.add_argument('--version',type=int);a=p.parse_args()
api=KaggleApi();api.authenticate()
s=json.loads(STATE.read_text()) if STATE.exists() else dict(kernel=KERNEL,competition=COMP)
def save():STATE.write_text(json.dumps(s,indent=2,default=str))
if a.action=='push':
 assert not s.get('push_intent'), 'Push intent exists: reconcile instead of duplicating'
 identity=json.loads((EVID/'release_identity.json').read_text())
 for name,sha in identity['release_files'].items():assert hashlib.sha256((ROOT/'release'/name).read_bytes()).hexdigest()==sha
 s['push_intent']=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity=identity);save()
 r=api.kernels_push(str(ROOT/'release'),timeout='32400',acc='NvidiaTeslaT4');d=json.loads(str(r));s['push_response']=d;save();print(r)
 assert d.get('ref','').removeprefix('/code/')==KERNEL
elif a.action=='status':
 r=api.kernels_status(KERNEL);s['kernel_status']=json.loads(str(r));save();print(r)
elif a.action=='submissions':
 rows=api.competition_submissions(COMP,page_size=100)
 (EVID/'competition_submissions.json').write_text(json.dumps([json.loads(str(x)) for x in rows],indent=2))
 for x in rows[:10]:print(x)
elif a.action=='submit':
 assert a.version is not None
 assert not s.get('submission_intent'), 'Submission intent exists: reconcile; never blindly retry'
 ev=json.loads((EVID/'platform_acceptance.json').read_text());assert ev['passed'] and ev['version']==a.version
 r=api.kernels_status(KERNEL);assert json.loads(str(r))['status']=='COMPLETE',r
 s['submission_intent']=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),version=a.version,acceptance=ev);save()
 r=api.competition_submit_code(file_name='submission.csv',message='Samanyu ICEBERG where GLACIER is blind public V1 reproduction; original pinned GPU image, RDKit 2026.03.3, fused top80 and fallback ICE; source-exact public12 checked; full hidden rerun with completeness assertions.',competition=COMP,kernel=KERNEL,kernel_version=a.version)
 s['submission_response']=json.loads(str(r));save();print(r)
