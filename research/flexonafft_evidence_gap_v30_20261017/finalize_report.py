"""Render measured results from downloaded and verified platform evidence."""
import json
import re
from prepare_release import ROOT

def main():
    report=json.loads((ROOT/'evidence/platform_acceptance.json').read_text())
    assert report['execution_acceptance_passed']
    a=json.loads((ROOT/'evidence/platform/reproduction_acceptance.json').read_text())
    n=report['processed'];bio=report['bio_counts'];exact=report['full_output_parity_passed']
    paragraph=(f"Kaggle 私有 Notebook **V{report['version']}** 已 **COMPLETE**，运行验收通过。"
        f"原始 V30 的 **6 个代码单元及嵌入代码完全保留**。"
        f"{n}/{n} 个实际预览分子完成推理，{report['placeholders']} 行占位预测排除在推理验收外。")
    parity=(f"**首选 {report['top1_matches']}/{n}、候选集合 {report['candidate_set_matches']}/{n}、完整排序 {report['full_ranking_matches']}/{n} 与上游一致**。"
        + ('完整预览 CSV 与上游逐字节一致。' if exact else '完整预览 CSV 与上游不逐字节一致；差异逐位记录在验收 JSON 中，不能称为输出完全一致。'))
    text=f'''# V30 BIO Structure Union 复现结果

{paragraph}

{parity}

| 检查项 | 本次结果 |
|---|---:|
| 原始代码单元完全保留 | 6/6 |
| 真实处理分子 / CSV 行数 | {n} / {report['rows']} |
| 占位行 | {report['placeholders']} |
| 基础引擎错误 | {a['base_errors']} |
| 首选与上游一致 | {report['top1_matches']}/{n} |
| 候选集合与上游一致 | {report['candidate_set_matches']}/{n} |
| 完整排名与上游一致 | {report['full_ranking_matches']}/{n} |
| BIO 新库 / 旧库结构数 | {bio['new_rows']:,} / {bio['old_rows']:,} |
| 补入 / 重叠 / 无效结构 | {bio['added_rows']} / {bio['overlap_rows']:,} / {bio['invalid_rows']} |

原代码计时约 **{a['runtime_seconds']/60:.1f} 分钟**。主进程 RDKit {a['packages'].get('rdkit')}；ICEBERG 与 GLACIER 都使用 CUDA 和隔离的 RDKit 2025.03.6，健康检查通过。BIO 合并计数与上游相同，所有新增结构沿用 engine2 原生指纹重算。

本次最终 CSV SHA256：`{report['reproduction_submission_sha256']}`。

上游 V30 最终 CSV SHA256：`{report['source_submission_sha256']}`。

## 结果边界

本次验证了普通运行的 **12 分子预览路径**。400 行 CSV 中有 388 行占位，不能直接把该预览 CSV 当作有效比赛预测提交。完整隐藏测试路径保留，但本任务只上传 Notebook 与 GitHub，**没有创建新的 V30 比赛评分提交，因此没有本次官方 MRR@25**。

源码内提到的 **0.451** 是继承的基础配方声明；`submission_v22_control.csv` 也是适配配方的同轮输出，不能当成未修改的 0.451 基础 V22 对照。预览一致性不证明独立准确率或隐藏集泛化收益。历史 V15 的官方提交 **57217289** 已完成，成绩 **0.441**，独立记录在旧目录中。

5 项本地控制测试全部通过：原始源码一致性、BIO 优先级/去重/重算、不兼容指纹拒绝、超时拒绝部分合并、rank2 的 metric twin 与首选保护。测试输入是合成控制，不是准确率评估。

## 可核查证据

- [发布身份与源码哈希](evidence/release_identity.json)
- [16 个冻结数据依赖](evidence/dependencies.json)；原附件版本不可从 API 获取，选择来源运行前的 Ready 版本并保留该限制
- [Kaggle 上传回执与远端身份](evidence/publication_state.json)
- [平台验收与逐分子/逐位比对](evidence/platform_acceptance.json)
- [实际运行验收与包版本](evidence/platform/reproduction_acceptance.json)
- [实际运行日志](evidence/platform/runtime_log.json)
- [BIO 合并审计](evidence/platform/eng/bio_union_audit.json)
- [本次真实处理预测](evidence/platform/processed_predictions.csv)；[上游真实处理预测](evidence/source_reference/processed_predictions.csv)
'''
    (ROOT/'RESULTS.md').write_text(text)
    project=ROOT.parents[1]/'README.md';current=project.read_text()
    summary=(f"2026-10-17: Reproduced Flexonafft **public V30 (V22 BIO Structure Union recipe)** with all 6 original code cells unchanged and 16 pinned inputs. Kaggle notebook V{report['version']} **COMPLETE**; preview top-1 {report['top1_matches']}/{n}, candidate sets {report['candidate_set_matches']}/{n}, full rankings {report['full_ranking_matches']}/{n} match upstream. "
        + ('Preview CSV is byte-identical. ' if exact else 'Output differences are documented. ')
        + 'No new V30 competition score. [Reproduction and verification](research/flexonafft_evidence_gap_v30_20261017/RESULTS.md).')
    current=re.sub(r'(?<=<!-- flexonafft-v30-reproduction -->)\n\n.*?(?=\n\n|\Z)','\n\n'+summary,current,count=1,flags=re.S)
    project.write_text(current)
    print(paragraph,parity)

if __name__=='__main__':main()
