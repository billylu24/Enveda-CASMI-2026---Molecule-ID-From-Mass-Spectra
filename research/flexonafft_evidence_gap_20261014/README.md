# Flexonafft「Closing the Evidence Gap」复现

来源：[Igor Zharov / flexonafft 的 Kaggle notebook](https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap)，公开 **V15**（`358431031`）。2026-10-14（美国洛杉矶时间）获取，算法为 **Iteration 07 — Replicated Fragment Evidence**。源码、嵌入模块和全部原作者署名保留；上游页面声明 Apache 2.0，数据及模型分别沿用各自许可。

本版本从 Initial 01 单独出发，在没有 GLACIER 且 ICEBERG 没有非 `None` 分数的分子中，对同分子式候选组增加碎片对比证据。组大小 2–8，分子最多 40 个键，至多两键断裂；每个有效谱至少有两个具有区分能力的峰，合计归一化强度至少 5%。准备后的强度和中性质量查询完全相同的谱只计算一次，至少 3 个不同有效谱才取候选得分中位数，以 1.25 权重重排并保护其他候选位置。不同谱并不保证统计独立。源码实际条件原样保留，包括针对 ICE 字典的 `is not None` 检查。

## 运行与验证

推理使用原始 Kaggle Python 3.13 / T4 x2 镜像，主进程 RDKit **2026.03.3**，模拟器隔离环境 RDKit **2025.03.6**，关闭网络。15 个数据依赖版本记录在 [dependencies.json](evidence/dependencies.json)。公开 API 没有提供原 notebook 的附件版本，选择其 `lastRunTime` 之前最新 Ready 数据版本，并记录这一限制。

从项目根目录运行（Kaggle CLI 须已登录、已能访问比赛）：

```bash
.venv/bin/python research/flexonafft_evidence_gap_20261014/prepare_release.py
.venv/bin/python -m unittest discover -s research/flexonafft_evidence_gap_20261014 -p 'test_*.py' -v
.venv/bin/python research/flexonafft_evidence_gap_20261014/publish.py push
.venv/bin/python research/flexonafft_evidence_gap_20261014/live_log.py --seconds 8
.venv/bin/python research/flexonafft_evidence_gap_20261014/publish.py status
.venv/bin/python research/flexonafft_evidence_gap_20261014/fetch_evidence.py run
.venv/bin/python research/flexonafft_evidence_gap_20261014/verify_run.py
```

宿主工具依赖见 `requirements-tools.txt`；它不替代冻结的推理镜像。`publish.py` 创建私有 Kaggle notebook，发布意图和版本保存在 `evidence/publication_state.json`，已有意图时阻止盲目重复发布。迁移至另一账户时，需同步修改发布/收集脚本与生成器中的 `KERNEL`。输出下载只选小型验收文件，禁止导出 signed URLs。

9 个原始代码单元逐字相同；只加署名 Markdown 和最终只读验收单元。`source/cell_*.py` 和 `source/embedded/` 是 `prepare_release.py` 从原 notebook 生成的可读镜像，实际执行仍使用 notebook 原单元。`release/reproduction.ipynb` 是可直接导入 Kaggle 的完整复现版本。

## 结果边界

原 notebook 的普通运行只实际处理 **12 个分子 / 41 条谱**，生成的 400 行 CSV 还有 **388 行占位符**。本次保留该行为；隐藏测试完整执行逻辑也原样保留。占位行不能算成已完成推理，也不能把整个预览 CSV 当作有效比赛预测提交。

上游 V15 预览：11 个 GLACIER 路由、1 个碎片回退路由；Iteration 07 考察 4 个候选组，接纳 0 个，改变 0 行。页面的 **0.441 Best Score 属于 V7**，不是 V15 或本次复现的实测分数。比赛提交需要显式运行下述 `submit` 操作；本次已按用户后续指令提交同一 V1。

**实测：**12/12 首选与候选集合一致，11/12 完整排序一致；`m_068354` 有 7 个低位排名不同，未达到逐字节输出一致。GPU 流程和同轮源码对照验收通过，跨运行差异定位于主引擎候选顺序，精确成因未确定。下载服务恢复后，两份 CSV 均已下载并核对各自哈希。

提交和状态查询（`submit` 会创建比赛提交，已有提交意图时会阻止重复）：

```bash
.venv/bin/python research/flexonafft_evidence_gap_20261014/publish.py submit
.venv/bin/python research/flexonafft_evidence_gap_20261014/submission_status.py
```

本次执行结果与验收详见 [RESULTS.md](RESULTS.md)。5 项本地检查覆盖源码和附件锁定、三谱门槛与去重、中位数、共享/弱证据拒绝、预算/键数限制和候选位置保护。它们不替代 GPU 完整流程验收或独立泛化评估。

已处理的 12 行预测另存为 `evidence/platform/processed_predictions.csv`；上游对应预测在 `evidence/source_reference/processed_predictions.csv`。这些文件不包含 388 行占位符。

<!-- official-competition-result -->

## 官方比赛提交

提交 **57217289**，notebook **V1**，状态 **PENDING**；当前尚无官方成绩。

在用户明确要求提交后，以同一 notebook V1 发起 **code submission**，Kaggle 在隐藏测试环境重新运行完整流程。此前 12 分子预览中的 388 行占位符没有作为直接 CSV 预测上传。完整结果以官方状态和成绩为准；可见预览与上游低位排名的差异仍保留在上述审计记录中。

官方返回见 [submission_result.json](evidence/submission_result.json)，提交意图和 API 回执见 [publication_state.json](evidence/publication_state.json)。
