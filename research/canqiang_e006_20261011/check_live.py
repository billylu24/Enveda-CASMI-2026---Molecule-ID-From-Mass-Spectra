"""Read live execution progress without publishing or submitting."""
import json
from pathlib import Path
from kaggle_evidence import api, KERNEL, EVID
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
client = api()
print(client.kernels_status(KERNEL))
with client.build_kaggle_client() as k:
    q = ApiListKernelSessionOutputRequest()
    q.user_name, q.kernel_slug = KERNEL.split('/')
    q.page_size = 100
    result = k.kernels.kernels_api_client.list_kernel_session_output(q)
(EVID / 'live_logs.json').write_text(result.log or '[]')
try:
    for item in json.loads(result.log or '[]')[-12:]:
        print(f"[{item.get('time',0):.1f}s] {item.get('data','')}", end='')
except ValueError:
    print((result.log or '')[-3000:])
