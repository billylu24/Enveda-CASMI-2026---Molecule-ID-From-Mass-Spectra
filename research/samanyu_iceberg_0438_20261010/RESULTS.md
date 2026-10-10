# Samanyu ICEBERG 实际复现结果

来源：`samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438` public V1。复现 notebook：`giaok246/casmi26-samanyu-iceberg-0-438-reproduction` V2。

Kaggle 提交 **57038936**，状态 **COMPLETE**，实际 public MRR@25 **0.438**。

上游标题为 0.438，本次相对标题差值 +0.000；提交前团队最高 0.433，相对差值 +0.005。

原始公开预览运行 993.78 秒，处理 12 分子；CSV 共 400 行，其中 388 行是原始预览占位符，未直接作为比赛预测上传。主引擎错误 0，第二引擎与 PubChem 各覆盖 12 个预览分子，12 行候选完整排名及整个预览 CSV 的 SHA256 与原 notebook 公开输出完全一致。

已处理预览每行 25–25 个有效且按原 metric identity 去重的 SMILES。保留原 top-80 融合，其中 1 个分子走 ICEBERG-only 回退。

主进程 RDKit 2026.03.3；模拟器 RDKit 2025.03.6。原权重、模型和阈值保留。完整源码修改范围和 SHA256 见 evidence/release_identity.json。

可见数据与训练数据重叠；预览检查不是泛化评估。提交的是固定 notebook 版本，而非预览 CSV。Kaggle 正式隐藏测试使用原全量预算，并执行新增的完整性断言；hidden 输出与逐阶段覆盖数字无法从公开 preview 下载，不作虚构报告。成绩来自官方隐藏测试评分。

| 模拟器 | 已评分分子 | 支持覆盖分子 | 状态 |
|---|---:|---:|---|
| ice | 11 | 11 | ok |
| glacier | 11 | 11 | ok |
