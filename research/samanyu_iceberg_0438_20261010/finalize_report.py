"""Record only official scored outcomes and the exact reproduction evidence."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'
state = json.loads((E / 'publication_state.json').read_text())
result = json.loads((E / 'submission_result.json').read_text())
acceptance = json.loads((E / 'platform_acceptance.json').read_text())['acceptance']
assert result['status'] == 'COMPLETE' and result['publicScore'] is not None
prior = json.loads((E / 'submissions_before.json').read_text())
best = max(float(x['publicScore']) for x in prior if x.get('publicScore'))
score = float(result['publicScore'])
summary = dict(source=acceptance['source'], source_version=1,
    notebook=state['kernel'], notebook_version=state['push_response']['versionNumber'],
    submission_id=result['ref'], status=result['status'], public_mrr25=score,
    upstream_title_score=0.438, delta_upstream_title=round(score-0.438, 6),
    prior_team_best=best, delta_prior_best=round(score-best, 6), visible_preview=acceptance,hidden_submission_complete=True)
(E / 'final_summary.json').write_text(json.dumps(summary, indent=2))
lines = ['# Samanyu ICEBERG 实际复现结果', '',
    f"来源：`{acceptance['source']}` public V1。复现 notebook：`{state['kernel']}` V{summary['notebook_version']}。", '',
    f"Kaggle 提交 **{result['ref']}**，状态 **{result['status']}**，实际 public MRR@25 **{score:.3f}**。", '',
    f"上游标题为 0.438，本次相对标题差值 {score-0.438:+.3f}；提交前团队最高 {best:.3f}，相对差值 {score-best:+.3f}。", '',
    f"原始公开预览运行 {acceptance['runtime_seconds']:.2f} 秒，处理 {acceptance['processed']} 分子；CSV 共 {acceptance['rows']} 行，其中 388 行是原始预览占位符，未直接作为比赛预测上传。主引擎错误 {acceptance['base_errors']}，第二引擎与 PubChem 各覆盖 {acceptance['processed']} 个预览分子，12 行候选完整排名及整个预览 CSV 的 SHA256 与原 notebook 公开输出完全一致。", '',
    f"已处理预览每行 {acceptance['candidate_min']}–{acceptance['candidate_max']} 个有效且按原 metric identity 去重的 SMILES。保留原 top-{acceptance['fusion_window']} 融合，其中 {acceptance['ice_only_fusion_molecules']} 个分子走 ICEBERG-only 回退。", '',
    f"主进程 RDKit {acceptance['rdkit']}；模拟器 RDKit 2025.03.6。原权重、模型和阈值保留。完整源码修改范围和 SHA256 见 evidence/release_identity.json。", '',
    '可见数据与训练数据重叠；预览检查不是泛化评估。提交的是固定 notebook 版本，而非预览 CSV。Kaggle 正式隐藏测试使用原全量预算，并执行新增的完整性断言；hidden 输出与逐阶段覆盖数字无法从公开 preview 下载，不作虚构报告。成绩来自官方隐藏测试评分。', '',
    '| 模拟器 | 已评分分子 | 支持覆盖分子 | 状态 |', '|---|---:|---:|---|']
for name, stage in acceptance['stages'].items():
    lines.append(f"| {name} | {stage['scored']} | {stage['covered']} | {stage['status']} |")
(ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
readme = ROOT / 'README.md'
body = readme.read_text().split('<!-- official-result -->')[0].rstrip()
readme.write_text(body + f"\n\n<!-- official-result -->\n实际提交 `{result['ref']}`，状态 `{result['status']}`，public MRR@25 **{score:.3f}**。详见 [RESULTS.md](RESULTS.md)。\n")
project_readme = ROOT.parents[1] / 'README.md'
project_body = project_readme.read_text()
lines = project_body.splitlines()
for i, line in enumerate(lines):
    if line.startswith('Best verified public MRR@25:') or line.startswith('Best public MRR@25:'):
        current_best = max(best, score)
        best_id = result['ref'] if score >= best else next(x['ref'] for x in prior if x.get('publicScore') and float(x['publicScore']) == best)
        lines[i] = f"Best verified public MRR@25: **{current_best:.3f}** (submission `{best_id}`, COMPLETE). Latest Samanyu ICEBERG V1 reproduction: **{score:.3f}**, submission `{result['ref']}`, COMPLETE. See [reproduction methods and official results](research/samanyu_iceberg_0438_20261010/RESULTS.md). The self-trained scientific research controller remains separate and in `local_only` mode; the historical snapshot below is retained."
        project_readme.write_text('\n'.join(lines) + '\n')
        break
else:
    raise ValueError('Results snapshot marker missing; reconcile README before GitHub push')
print(json.dumps(summary, ensure_ascii=False, indent=2))
