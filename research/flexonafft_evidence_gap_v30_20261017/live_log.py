"""Read a short current Kaggle log tail without changing the run."""
import json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
from prepare_release import ROOT,KERNEL
api=KaggleApi();api.authenticate();print(api.kernels_status(KERNEL))
with api.build_kaggle_client() as k:
    req=ApiListKernelSessionOutputRequest();req.user_name,req.kernel_slug=KERNEL.split('/');req.page_size=1
    response=k.kernels.kernels_api_client.list_kernel_session_output(req)
(ROOT/'evidence/live_logs.json').write_text(response.log or '[]')
for row in json.loads(response.log or '[]')[-12:]:
    print(f"[{row.get('time',0):.1f}s] {row.get('data','')}",end='')
