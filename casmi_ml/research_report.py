"""Render measured research artifacts to a concise, source-linked Markdown report."""

import argparse
import json
from pathlib import Path

from casmi_ml.research_protocol import ROOT


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def report(root=ROOT, output=None):
    root = Path(root)
    output = Path(output or root / "REPORT.md")
    chemical = read(root / "chemical_selection.json")
    acceptance = read(root / "chemical_acceptance.json")
    representation = read(root / "representation_selection.json")
    generation = read(root / "generation/development_report.json")
    generation_holdout = read(root / "generation/holdout_pilot_report.json")
    budget = read(root / "training_budget.json")
    lines = [
        "# CASMI 化学证据与生成首轮结果",
        "",
        "以下为本地代理实验，不能等同于 Kaggle 成绩。未自动上传、提交或替换历史生产方案。",
        "",
        "## 化学重排序",
        "",
    ]
    if chemical:
        lines += ["| 配置 | 未知 MRR | 已知 MRR | 开发合格 |", "|---|---:|---:|---|"]
        lines.append(
            f"| 冻结基线 | {chemical['baseline']['unknown']['mrr25']:.6f} | {chemical['baseline']['known']['mrr25']:.6f} | — |"
        )
        for r in chemical["results"]:
            lines.append(
                f"| {r['component']} 权重 {r['weight']} | {r['reports']['unknown']['mrr25']:.6f} | {r['reports']['known']['mrr25']:.6f} | {r['eligible']} |"
            )
    else:
        lines += ["尚未完成开发消融。"]
    lines += ["", "## 独立验收", ""]
    if acceptance and acceptance.get("holdout_opened"):
        lines += [
            "| 场景 | 分子数 | 基线 MRR | 新 MRR | 差值 95% 区间 |",
            "|---|---:|---:|---:|---|",
        ]
        for m, r in acceptance["reports"].items():
            ci = r["paired"]["ci95"]
            lines.append(
                f"| {m} | {r['selected']['molecules']} | {r['baseline']['mrr25']:.6f} | {r['selected']['mrr25']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |"
            )
        lines += [
            "",
            f"化学方案独立验收通过：**{acceptance['accepted']}**。已知保护采用点估计门槛，不能当作统计非劣证明。",
        ]
    else:
        lines += ["尚未打开或完成化学独立验收。"]
    lines += [
        "",
        "## 表征对照",
        "",
        "| 方法 | 最佳合格未知 MRR | 微调轮数 | 开发改善 |",
        "|---|---:|---:|---|",
    ]
    for method in ["supervised", "masked", "dino"]:
        r = read(root / "representation" / f"{method}_42_v2/result.json")
        if r:
            best = r["best_routed_mrr"]
            score = f"{best:.6f}" if best >= 0 else "没有合格 checkpoint"
            lines.append(
                f"| {method} | {score} | {sum(h['phase'] == 'finetune' for h in r['history'])} | {r['accepted_for_holdout']} |"
            )
        else:
            lines.append(f"| {method} | 未完成 | — | — |")
    if representation:
        lines += [
            "",
            f"表征选择是否进入独立验收：{representation['accepted_for_holdout']}。",
        ]
    lines += ["", "## 结构生成", ""]
    if generation:
        lines += [
            f"开发试验 {generation['molecules']} 个分子；这是有限样本可行性检查，不是完整独立验收。",
            f"生成有效率 {generation['valid_rate']:.2%}，质量匹配率 {generation['mass_match_rate']:.2%}，预测分子式 Top-5 {generation['formula_top5']:.2%}。",
            f"纯生成 MRR {generation['pure']['mrr25']:.6f}；合并检索 MRR {generation['merged']['mrr25']:.6f}；同队列基线 {generation['baseline']['mrr25']:.6f}。",
            f"候选池外新增精确答案 {generation['new_exact_truths_outside_retrieval_pool']} 个。合法性及 Tanimoto 相似度不能替代精确结构命中。",
        ]
    else:
        lines += ["生成训练或开发采样尚未完成。"]
    if generation_holdout:
        r = generation_holdout
        lines += [
            "",
            f"留出可行性检查 {r['molecules']} 分子：纯生成 MRR {r['pure']['mrr25']:.6f}，合并 {r['merged']['mrr25']:.6f}，基线 {r['baseline']['mrr25']:.6f}；候选池外精确答案 {r['new_exact_truths_outside_retrieval_pool']} 个。",
            "该子集未进行完整独立验收；生成模型尚无部署资格。",
        ]
    lines += ["", "## 资源与局限", ""]
    if budget:
        for stage, r in budget.items():
            lines.append(
                f"- {stage} 累计训练 {r['used_seconds'] / 3600:.2f} 小时，上限 {r['limit_seconds'] / 3600:.0f} 小时。"
            )
    lines += [
        "- 候选目录缺失仍限制检索；碎裂评分有误差，对不支持的离子/仪器跳过。",
        "- 当前 0.75/0.25 RRF 在检索列表至少 25 项时，无法把仅生成的新候选送入前 25 名；下一轮须开发新的合并规则。",
        "- GAN 本轮仅完成可行性设计；大分子、离散生成与谱图条件匹配使其优先级低于受约束自回归基线。",
        "- 详细实现与复现命令见仓库 docs/RESEARCH_20261001.md；原始分子级结果与冻结 JSON 位于本报告同目录。",
        "",
    ]
    output.write_text("\n".join(lines))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    print(report(a.root, a.output))


if __name__ == "__main__":
    main()
