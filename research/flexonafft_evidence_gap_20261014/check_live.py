"""Read current run status and a short log tail without changing the run."""
import json
from fetch_evidence import client, KERNEL, EVID
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
api = client()
print(api.kernels_status(KERNEL))
with api.build_kaggle_client() as k:
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = KERNEL.split('/'); req.page_size = 1
    result = k.kernels.kernels_api_client.list_kernel_session_output(req)
(EVID / 'live_logs.json').write_text(result.log or '[]')
for item in json.loads(result.log or '[]')[-8:]:
    print(f"[{item.get('time', 0):.1f}s] {item.get('data', '')}", end='')
