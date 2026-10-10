"""Read the documented Kaggle REST session log stream, omitting signed endpoint details."""
import argparse,json,time,requests
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=15);p.add_argument('--session',type=int,default=357004551);p.add_argument('--label',default='live_logs_v1');a=p.parse_args()
assert all(c.isalnum() or c in '_-' for c in a.label)
root=Path(__file__).resolve().parent
api=KaggleApi();api.authenticate();rows=[];start=time.time()
with api.build_kaggle_client() as k:
 c=k._http_client;c._init_session()
 try:
  with c._session.get(f'https://www.kaggle.com/api/v1/kernels/sessions/{a.session}/logs/stream',params={'waitForLogsUrlSeconds':1},stream=True,timeout=(10,5)) as r:
   r.raise_for_status()
   for line in r.iter_lines(chunk_size=1):
    if line.startswith(b'data: '):
     try:rows.append(json.loads(line[6:]))
     except ValueError:pass
    if time.time()-start>a.seconds:break
 except requests.RequestException as e:
  if not rows:print('LIVE_LOG_UNAVAILABLE',type(e).__name__)
if rows:
 p=root/'evidence'/f'{a.label}.json'
 previous=json.loads(p.read_text()) if p.exists() else []
 last=max((x.get('time',0) for x in previous),default=-1)
 new=[x for x in rows if x.get('time',0)>last]
 p.write_text(json.dumps(rows))
 readable=''.join(f"[{x.get('time',0):.1f}s] {x.get('data','')}" for x in rows)
 (root/'evidence'/f'{a.label}.log').write_text(readable)
 print('NEW_LOG_ROWS',len(new),'LATEST_TIME',max(x.get('time',0) for x in rows))
 if new:print('LATEST', ''.join(f"[{x.get('time',0):.1f}s] {x.get('data','')}" for x in new)[-4500:])
