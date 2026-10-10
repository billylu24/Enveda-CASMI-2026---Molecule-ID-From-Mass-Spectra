"""Read current platform execution logs without publishing or submitting."""
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
ROOT = Path(__file__).resolve().parent
api = KaggleApi(); api.authenticate()
state = json.loads((ROOT/'evidence/publication_state.json').read_text())
print(api.kernels_status(state['kernel']))
owner, slug = state['kernel'].split('/')
with api.build_kaggle_client() as client:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = owner; req.kernel_slug = slug; req.page_size = 100
    result = client.kernels.kernels_api_client.list_kernel_session_output(req)
    if result.log:
        (ROOT/'evidence/live_pipeline.log').write_text(result.log)
        records = json.loads(result.log)
        text = ''.join(f"[{row.get('time', 0):.1f}s] {row.get('data', '')}" for row in records)
        (ROOT/'evidence/live_pipeline_readable.log').write_text(text)
        print(text[-4500:])
