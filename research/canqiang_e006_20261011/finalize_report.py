"""Write the official score without substituting any upstream claim."""
import argparse
import json
from pathlib import Path
from kaggle_evidence import ROOT, EVID, KERNEL
parser = argparse.ArgumentParser()
parser.add_argument('--pending', action='store_true')
args = parser.parse_args()
record = json.loads((EVID / 'submission_result.json').read_text())
acceptance = json.loads((EVID / 'platform_acceptance.json').read_text())
state = json.loads((EVID / 'publication_state.json').read_text())
assert record['status'] in ({'PENDING', 'RUNNING'} if args.pending else {'COMPLETE'})
assert args.pending or record.get('publicScore') is not None
score = None if args.pending else float(record['publicScore'])
result_line = f"本次提交 **{record['ref']}**，状态 **{record['status']}**，" + ('完整隐藏测试正在运行，尚无官方成绩。' if args.pending else f'public MRR@25 **{score:.3f}**。')
result_en = (f"Official code submission **{record['ref']}**, notebook V{state['push_response']['versionNumber']}, is **{record['status']}**; the full hidden test is running and no score is available yet." if args.pending else f"Official code submission **{record['ref']}**, notebook V{state['push_response']['versionNumber']}, completed with public MRR@25 **{score:.3f}**.")
summary = dict(source='canqiang/casmi26-e006-window-40-ice-fallback-true', source_version=1,
    kernel=KERNEL, notebook_version=state['push_response']['versionNumber'],
    submission_id=record['ref'], status=record['status'], public_mrr25=score,
    source_code_exact=True, original_code_cells=30, source_preview_exact=True,
    processed_preview_molecules=acceptance['acceptance']['processed'], preview_placeholders_excluded=388,
    window=40, ice_fallback=True, upstream_page_public_score=0.440,
    delta_upstream_page=None if args.pending else round(score - 0.440, 6),
    note='Official code-submission score; visible preview is not independent validation.')
(EVID / ('pending_summary.json' if args.pending else 'final_summary.json')).write_text(json.dumps(summary, indent=2))
url=f'https://www.kaggle.com/code/{KERNEL}'
result=f'''# E006 官方结果

{result_line}

- 原始来源：Canqiang E006 公开 V1，30 个代码单元保持一致。
- 复现 notebook：[Kaggle V{summary['notebook_version']}]({url})。
- 处理：窗口 40、ICE fallback=True，全部原模型和数据版本保留。
- 预览：12 个处理分子的完整排序与上游一致，400 行 CSV 字节 SHA256 一致；388 个占位行未直接提交。
- 比赛：通过同一 notebook V{summary['notebook_version']} 的 code submission 运行完整隐藏测试，{'当前等待运行和评分；最终状态以官方返回为准。' if args.pending else '完整处理断言通过后返回成绩。'}

验收见 `evidence/platform_acceptance.json` 和 `evidence/source_preview_comparison.json`；官方返回见 `evidence/submission_result.json`。可见预览不是独立验证，官方 public 成绩也不等于 private 成绩。
'''
(ROOT / 'RESULTS.md').write_text(result)
p=ROOT/'README.md';s=p.read_text();s=s.split('<!-- official-result -->')[0]+f'<!-- official-result -->\n{result_line} 详见 [RESULTS.md](RESULTS.md)。\n';p.write_text(s)
project=ROOT.parents[1];p=project/'README.md';s=p.read_text();marker='<!-- canqiang-e006-reproduction -->';section=f'''{marker}

### Canqiang E006 reproduction (2026-10-11)

Reproduced public V1 with all 30 original code cells unchanged, the original Python 3.12 T4 image, 16 pinned datasets, candidate window 40 and ICEBERG fallback enabled. The 12-molecule preview CSV matches the source exactly; placeholders were excluded from direct submission. {result_en} [Reproduction and evidence](research/canqiang_e006_20261011/README.md); [official result](research/canqiang_e006_20261011/RESULTS.md).
''';s=s.split(marker)[0].rstrip()+'\n\n'+section;p.write_text(s)
print(json.dumps(summary, indent=2))
