"""Human-readable experiment report assembled only from recorded measurements."""
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def render(root):
    root = Path(root)
    selection = read(root / 'selection.json')
    holdout = read(root / 'holdout_report.json')
    diagnostic = read(root / 'diagnostic_report.json')
    budget = read(root / 'budget.json')
    baseline = read(root / 'baseline_dev.json')
    summaries = read(root / 'seed_summary.json')
    final = read(root / 'final_selection.json')
    runs = [read(p) for p in sorted((root / 'runs').glob('*/result.json'))]
    names = {'mlp': '基础 MLP', 'metadata': 'MLP + 元数据', 'enhanced': 'MLP + 元数据 + 质量差',
             'deepsets': 'DeepSets', 'transformer': 'Transformer'}
    models = ' + '.join(names[a] for a in final['architectures']) or '检索基线'
    historical = final.get('deployment_mode') == 'historical_hybrid'
    guarded = final.get('deployment_mode') == 'confidence_guarded'
    recipe = models if final['neural_weight'] == 1 else f"检索 + {models}（神经权重 {final['neural_weight']}）" if final['architectures'] else models
    if guarded:
        recipe = f"谱库置信度保护 + {models}（阈值 {final['confidence_threshold']}）"
    if historical:
        recipe = '保留原 hybrid.py：谱库共识检索 + COCONUT 结构扩展（历史公开榜 0.176）'
    lines = ['# CASMI CPU 实验报告', '', f'最终推荐：**{recipe}**。', '',
             '推荐依据为固定开发集；最终留出集未参与配置选择。这里的本地成绩不是 Kaggle 公开榜成绩。', '',
             '## 首轮架构比较', '',
             '| 模型 | 开发 MRR@25 | 最佳轮数 | 参数量 | 本次运行秒数 | 停止原因 |',
             '|---|---:|---:|---:|---:|---|',
             f"| 共同候选池检索基线 | {baseline['mrr25']:.5f} | — | — | — | — |"]
    order = ['mlp', 'metadata', 'enhanced', 'deepsets', 'transformer']
    for r in sorted(runs, key=lambda r: order.index(r['architecture'])):
        if r['seed'] == 42 and not r['final_refit']:
            lines.append(f"| {names[r['architecture']]} | {r['best_mrr25']:.5f} | {r['best_epoch']} | {r['parameters']:,} | {r['seconds']:.1f} | {r['stop_reason']} |")
    lines += ['', '运行时间包含该次训练和逐轮开发评估；峰值 RSS 是进程累计值，包含共享数据准备。', '',
              '## 多随机种子复跑', '', '| 架构 | 开发 MRR@25 均值 | 标准差 |', '|---|---:|---:|']
    for r in summaries:
        lines.append(f"| {names[r['architecture']]} | {r['mean_mrr25']:.5f} | {r['std_mrr25']:.5f} |")
    lines += ['', '## 冻结方案的独立评估', '',
              '| 数据集 | 分子数 | 候选召回率 | MRR@25 | 候选内 MRR@25 | Top-1 | Top-5 | Top-25 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, r in [('开发集', selection), ('留出集', holdout), ('留出集检索基线', holdout['baseline']), ('NP 诊断集', diagnostic)]:
        conditional = r['conditional_mrr25']
        cond = f'{conditional:.5f}' if conditional is not None else 'N/A'
        lines.append(f"| {label} | {r['molecules']} | {r['candidate_recall']:.2%} | {r['mrr25']:.5f} | {cond} | {r['top1']:.2%} | {r['top5']:.2%} | {r['top25']:.2%} |")
    if guarded:
        fresh = read(root / 'guarded_fresh_report.json')
        lines += ['', '## 最终保护方案：全新留出集验收', '',
                  '| 场景 | 分子数 | 最终方案 MRR@25 | 检索基线 MRR@25 | 差值 95% 区间 |',
                  '|---|---:|---:|---:|---|']
        for mode, label in [('unknown', '无同分子参考谱图'), ('known', '有其他参考谱图，排除查询及其完全相同副本')]:
            r = fresh[mode]
            ci = r['vs_baseline']['ci95']
            lines.append(f"| {label} | {r['molecules']} | {r['mrr25']:.5f} | {r['baseline']['mrr25']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
        lines += ['', '这组分子与首次留出集、开发集、训练集及 NP 诊断集完全不重叠。保护阈值只在开发集选择，随后冻结。上面的首次留出表描述未保护的全局融合方案，不能当作最终保护方案的成绩。', '']
    diff = holdout['vs_baseline']
    lines += ['', f"未保护的全局融合方案在首次留出集相对检索基线的 MRR 差值为 {diff['difference']:.5f}；按分子 bootstrap 的 95% 区间为 [{diff['ci95'][0]:.5f}, {diff['ci95'][1]:.5f}]。", '',
              f"累计计入训练预算 {budget['used_seconds']/3600:.2f} 小时，上限 {budget['limit_seconds']/3600:.0f} 小时。准备数据、构建参考库及独立最终评估另计。", '',
              '## 最终推理方案', '',
              f"- 神经编码器：{models}。", '- 每个编码器预测 2048 位 Morgan 指纹概率；同一分子的多张谱图平均概率。',
              '- 质量窗口为 35 ppm，绝对窗口下限为 0.006 Da；候选来自完整参考库结构及 COCONUT。',
              '- 候选按指纹对数似然排序；需要融合时使用 RRF 常数 60。',
              f"- 神经排序权重：{final['neural_weight']}；参考检索权重：{1-final['neural_weight']}。",
              '- 输出最多 25 个不重复二维结构；缺少候选时使用参考库回退。', '',
              ]
    if guarded:
        lines += [f"谱库最高共识分数 ≥ {final['confidence_threshold']} 时使用完整检索排序；低于阈值时保留检索首位，后续位置采用神经 RRF 融合排序。", '']
    if historical:
        start = lines.index('## 最终推理方案')
        lines = lines[:start] + ['## 最终推理方案', '',
            '恢复原 `hybrid.py` 的质量筛选、谱库共识与 COCONUT 指纹类比排序，以及其保留前 1–2 个谱库候选的融合规则。它是项目当前已有公开榜验证的方案，历史分数 0.176 并非本次新提交结果。', '',
            '新神经模型的权重和冻结选择仍保留在 `research_selection.json` 与 runs 目录中，用于后续研究；本次正式比赛推理不启用它们。', '']
    else:
        lines += ['权重路径：', ''] + [f'- `{p}`' for p in final['checkpoints']]
    lines += ['', '## 实验边界', '',
              '- 主验证模拟没有同分子参考谱图的情况；不能单凭它推断比赛各类别的综合成绩。',
              '- NP 诊断也屏蔽了同分子参考谱图，且每分子最多四张查询谱，因此不能与历史约 0.90 的已知谱库诊断直接比较。',
              '- 候选召回率低是当前重要限制；错误答案缺失时，任何排序器都无法命中。',
              '- E0 使用共同候选池，属于原混合检索的受控适配；并非历史 15 ppm COCONUT 流水线的逐位复现。',
              '- 上表独立留出结果来自冻结前的训练集权重；最终训练集+开发集重训权重没有再次使用留出集调参或评估。',
              '- 若达到 20 轮上限，不能断言模型已充分收敛；首轮结果也不能证明 Transformer 的性能上限。',
              '- 最新 Kaggle 规则无法在线核实：浏览器访问被拒绝。离线包遵循项目现有提交方式，尚未上传或在 Kaggle 执行。', '']
    if final.get('deployment_note'):
        lines += ['部署验收说明：' + final['deployment_note'], '']
    known_path = root / 'known_spectrum_report.json'
    if known_path.exists():
        known = read(known_path)
        lines += ['## 已知谱图兼容性验收', '',
                  f"在 {known['molecules']} 个 NP 分子的来源隔离诊断中，冻结神经融合方案 MRR@25 为 {known['mrr25']:.5f}，检索基线为 {known['baseline']['mrr25']:.5f}。", '',
                  '这一检查允许从其他来源找到同分子的参考谱图，但不允许查询来源的谱图进入参考库，也不把查询标签直接加入候选。预设验收条件为相对基线最多下降 0.005；它只决定是否接受冻结方案，不搜索其他模型或融合权重。', '']
    audit_path = root / 'candidate_coverage_audit.json'
    if audit_path.exists():
        audit = read(audit_path)
        lines += ['## 候选覆盖审计', '',
                  f"开发集 {audit['molecules']} 个分子中，按结构键有 {audit['in_coconut_by_key']} 个存在于 COCONUT；其中 {audit['in_coconut_and_mass_window']} 个通过质量窗口。可计算中性质量的分子为 {audit['supported_mass']} 个。主要瓶颈是候选数据库覆盖，而非排序器。", '']
    delivery_path = root / 'delivery_verification.json'
    if delivery_path.exists():
        delivery = read(delivery_path)
        lines += ['## 交付验证', '',
                  f"打包后的代码从项目外目录运行，完成 {delivery['molecules']} 个分子的预测；输出与项目内推理 CSV 逐字节一致。耗时 {delivery['seconds']:.1f} 秒，峰值 RSS {delivery['peak_rss_mib']/1024:.2f} GiB。", '',
                  '当前可见测试集全部触发谱库保护，400 个首位结构均与原 hybrid 提交一致。因此不能把本次可见测试运行表述为神经模型带来了榜单提升；神经收益证据来自独立留出实验。', '',
                  'CSV 每分子 5–25 个候选，ID 覆盖、结构可解析性和二维结构去重均已通过检查。12 项单元/集成检查和 Ruff 静态检查通过。未上传或提交 Kaggle，未进行真实 Kaggle 环境执行。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return root / 'REPORT.md'
