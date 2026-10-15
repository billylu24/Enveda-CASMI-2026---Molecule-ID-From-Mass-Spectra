"""Render the measured execution result from saved acceptance evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVID = ROOT / 'evidence'


def main():
    checked = json.loads((EVID / 'platform_acceptance.json').read_text())
    assert checked['passed'] and checked['completed_gpu_run']
    run = json.loads((EVID / 'platform/reproduction_acceptance.json').read_text())
    source = json.loads((EVID / 'source_reference/research_routing_audit.json').read_text())
    measured = run['initial01']; iteration = run['iteration07']
    comparison = checked['source_preview_comparison']
    transport = ('上游和本次 CSV 下载成功，文件字节与运行时哈希均已核对。'
                 if checked['downloaded_csv_bytes_verified'] else
                 'Kaggle 输出文件下载服务超时；报告取自认证 API 返回的完整运行日志。'
                 '已比较两个 GPU 进程实际计算并打印的 CSV SHA256，尚未独立下载两份 CSV 做本地字节比较。'
                 '各文件来源和下载状态记录在 evidence_transport.json。')
    text = f'''# Flexonafft V15 / Iteration 07 复现结果

本次私有 Kaggle notebook **V{checked['version']}** 已 **COMPLETE**，最终只读验收通过。原始 V15 的 9 个代码单元与平台实际运行源码完全相同。本次没有比赛提交，也没有本次 public MRR@25。

| 检查 | 上游 V15 预览 | 本次复现 |
|---|---:|---:|
| 实际处理分子 | 12 | {run['processed']} |
| CSV 行数 | 400 | {run['rows']} |
| 占位行（排除在推理验收外） | 388 | {run['placeholders']} |
| GLACIER 路由 | {source['routes'].get('glacier', 0)} | {measured['routes'].get('glacier', 0)} |
| 原碎片回退路由 | {source['routes'].get('frag_gap', 0)} | {measured['routes'].get('frag_gap', 0)} |
| Iteration 07 考察组 | 4 | {iteration['details']['target_groups']} |
| Iteration 07 接纳组 | 0 | {iteration['details']['accepted_groups']} |
| 相对 Initial 01 改变列表 / 首选 | 0 / 0 | {iteration['changed_rows']} / {iteration['changed_top1']} |
| 基础引擎错误 | 0 | {run['base_errors']} |

本次耗时约 **{run['runtime_seconds']/60:.1f} 分钟**（原代码计时）。主进程 RDKit {run['main_rdkit']}，ICEBERG 和 GLACIER 都使用 CUDA 与 RDKit 2025.03.6；模拟器健康检查及原代码同轮路由重建断言通过。实际包版本保存在运行验收 JSON 中。

本次最终 CSV SHA256：

```
{checked['submission_sha256']}
```

上游最终 CSV SHA256：

```
{comparison['source_submission_sha256']}
```

{transport}

## 与上游预览的差异

逐行下载比对显示：**12/12 首选一致，12/12 候选集合一致，11/12 完整排序一致**。仅 `m_068354` 的第 **11、12、13、19、20、24、25** 位候选顺序不同；没有新增或删除其最终 25 个候选。两份最终 CSV **不是逐字节一致**，`full_output_parity_passed=false`；运行验收通过不代表跨运行输出一致性通过。

下载的同轮候选快照将差异定位到主引擎 `prefusion` / `base_tail` 顺序；Engine2 和 PubChem 的全部 12 个分子输入完全一致。ICE 输入候选数为上游 773 / 本次 772，最终打分候选为 755 / 754。差异在 Iteration 07 前已存在；精确数值成因尚未确定，不将其未经验证地归因于 GPU 随机性。原代码及算法未为了对齐输出而改动。完整逐位差异和两侧 SMILES 保存在 [source_preview_comparison.json](evidence/source_preview_comparison.json)。

Iteration 07 在这个可见子集接纳 0 组，因此结果与 Initial 01 相同。这不能证明隐藏集也无变化，或该方法提高了识别准确率。5 项本地测试额外验证了去重、三谱门槛、中位数及候选位置保护等关键行为；测试输入是合成控制，不是准确率评估。

页面展示的 **0.441 Best Score 属于 V7（357977484）**。当前 V15 的历史成绩没有验证，本次也没有新的官方分数；不将历史最佳分数写成复现成绩。400 行预览文件含 388 行占位符，不能直接用于比赛预测提交。完整隐藏测试逻辑保留在代码中，但本任务仅执行了原 notebook 的 12 分子预览路径。

证据：

- [源码与发布身份](evidence/release_identity.json)
- [来源页面观察](evidence/source_page_evidence.json)
- [平台验收与跨运行比对](evidence/platform_acceptance.json)
- [实际运行验收 JSON](evidence/platform/reproduction_acceptance.json)
- [认证 API 运行日志](evidence/platform/runtime_log.json)
- [上游 Iteration 07 审计](evidence/source_reference/iteration_audit.json)
- [本次 Iteration 07 审计](evidence/platform/iteration_audit.json)

上游日志内保留的 E004/旧版本标识属于继承的观察器，不应解读为本次最终 notebook 的版本。最终来源身份由 V15 源码快照、API metadata 与 release_identity.json 共同记录。
'''
    (ROOT / 'RESULTS.md').write_text(text)
    project = ROOT.parents[1] / 'README.md'
    current = project.read_text()
    marker = '<!-- flexonafft-v15-reproduction -->'
    addition = f'''{marker}

Flexonafft **Closing the Evidence Gap V15 / Iteration 07** reproduction completed on T4 with all 9 original code cells unchanged: 12 processed molecules, 388 preview placeholders, 12/12 matching top-1 and candidate sets, 11/12 matching complete rankings, and 0 Iteration 07 changed rows. One lower-rank ordering differs from upstream and is documented. No competition submission or new score. See [reproduction and measured execution results](research/flexonafft_evidence_gap_20261014/RESULTS.md).
'''
    if marker not in current:
        first, rest = current.split('\n\n', 1)
        project.write_text(first + '\n\n' + addition + '\n' + rest)
    print(text)

if __name__ == '__main__': main()
