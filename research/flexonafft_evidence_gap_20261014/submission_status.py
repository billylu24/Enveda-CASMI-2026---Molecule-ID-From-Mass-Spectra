"""Read this exact competition submission and update its report without resubmitting."""
import datetime
import json
import re
from publish import ROOT, EVID, KERNEL, COMPETITION
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    state = json.loads((EVID / 'publication_state.json').read_text())
    submission_id = state['submission_response']['ref']
    api = KaggleApi(); api.authenticate()
    records = [json.loads(str(x)) for x in api.competition_submissions(COMPETITION, page_size=100)]
    matches = [x for x in records if str(x['ref']) == str(submission_id)]
    assert len(matches) == 1, 'Submission missing from recent history; reconcile explicitly'
    record = matches[0]
    assert record['url'].split('?')[0].removeprefix('/code/') == KERNEL
    (EVID / 'submission_result.json').write_text(json.dumps(record, indent=2))
    score = float(record['publicScore']) if record.get('publicScore') is not None else None
    result = dict(submission_id=submission_id, status=record['status'], public_mrr25=score,
        kernel=KERNEL, notebook_version=state['push_response']['versionNumber'],
        source_version=15, code_submission=True, visible_preview_processed=12,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (EVID / 'competition_result.json').write_text(json.dumps(result, indent=2))
    measured = f'官方 public MRR@25 **{score:.3f}**。' if score is not None else '当前尚无官方成绩。'
    summary = f"提交 **{submission_id}**，notebook **V{result['notebook_version']}**，状态 **{record['status']}**；{measured}"
    marker = '<!-- official-competition-result -->'
    section = f'''{marker}

## 官方比赛提交

{summary}

在用户明确要求提交后，以同一 notebook V1 发起 **code submission**，Kaggle 在隐藏测试环境重新运行完整流程。此前 12 分子预览中的 388 行占位符没有作为直接 CSV 预测上传。完整结果以官方状态和成绩为准；可见预览与上游低位排名的差异仍保留在上述审计记录中。

官方返回见 [submission_result.json](evidence/submission_result.json)，提交意图和 API 回执见 [publication_state.json](evidence/publication_state.json)。
'''
    for name in ['RESULTS.md', 'README.md']:
        path = ROOT / name
        text = path.read_text().split(marker)[0].rstrip()
        text = text.replace('本次没有比赛提交，也没有本次 public MRR@25。',
                            '本文前半部分记录可见预览验收；后续官方比赛提交见文末。')
        path.write_text(text + '\n\n' + section)
    project = ROOT.parents[1] / 'README.md'
    text = project.read_text()
    text = re.sub(r'(?<=<!-- flexonafft-v15-reproduction -->)\n\n.*?(?=\n\n|\Z)',
        f"\n\nFlexonafft **Closing the Evidence Gap V15 / Iteration 07** reproduced with all 9 code cells unchanged. T4 preview: 12/12 matching top-1 and candidate sets, 11/12 matching complete rankings; one lower-rank ordering difference is documented. Official code submission **{submission_id}**, notebook V1, is **{record['status']}**" +
        (f" with public MRR@25 **{score:.3f}**." if score is not None else '; no official score yet.') +
        ' [Reproduction and official result](research/flexonafft_evidence_gap_20261014/RESULTS.md).', text, count=1, flags=re.S)
    project.write_text(text)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
