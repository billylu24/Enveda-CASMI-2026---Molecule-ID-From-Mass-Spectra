# Samanyu ICEBERG 复现与提交记录

已提交 Kaggle，submission **57038936**，状态 **PENDING**（2026-10-10 UTC）。固定 notebook：`giaok246/casmi26-samanyu-iceberg-0-438-reproduction` V2，scriptVersionId `357007287`。

实际 public MRR@25 尚未返回；上游标题 0.438 不是本次成绩。

全部原代码单元逐字一致。公开预览处理 12 分子，候选完整排序 **12/12** 与上游一致；整个预览 CSV SHA256 同为 `880e9c85572fb8162d1fc6b0e34f0c147f8088e88845504a6fb4ed259cb57f81`。

预览运行 993.78 秒，主引擎错误 0，第二引擎与 PubChem 各处理 12 分子。ICEBERG 与 GLACIER 各覆盖并评分 11 分子，预测错误为 0。每个已处理分子有 25 个有效、按原 metric identity 去重的候选。

提交的是固定 notebook 版本，正式 hidden rerun 运行原始全量预算及新增完整性断言。没有直接上传包含 388 行预览占位符的 CSV。可见预览与训练集重叠，仅用作执行和排序检查。

`monitor.py` 正在轮询官方结果，完成后更新此报告、项目 README 并推送 GitHub。详见 `evidence/publication_state.json`、`evidence/platform_acceptance.json` 和 `evidence/source_preview_comparison.json`。
