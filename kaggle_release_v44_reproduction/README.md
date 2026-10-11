# CASMI26 V44 PairTail Locked Top1 — 0.417 复现归档

归档日期：2026-10-05（America/Los_Angeles）  
状态：Kaggle competition submission **Succeeded**，Public MRR@25 **0.417**。

## 权威对象

- 原作者 Notebook：<https://www.kaggle.com/code/imranarif536/casmi26-v44-pairtail-locked-top1>
- 本账户精确复现：<https://www.kaggle.com/code/xiaoyuzhoux120/casmi26-v44-pairtail-locked-top1?scriptVersionId=355245043>
- Kaggle 版本标题：`Reproduce original V3 354936340`
- 提交描述：`V44 PairTail Locked Top1 — exact reproduction of public V3 (354936340).`
- Kaggle 运行：GPU T4 x2，`2h 2m 51s`；日志计时 `7370.8 s`，successful。
- Competition submission：Succeeded，Public Score `0.417`。
- 最终提交选择：核验时页面为 `0/2`，未手动选择；少于两个时 Kaggle 会在截止时自动选择最佳成绩。

## 本地冻结文件

| 文件 | 用途 | SHA256 |
|---|---|---|
| `source/casmi26-v44-pairtail-locked-top1-v355245043.ipynb` | Kaggle 下载的完整 Notebook，含输出 | `a2e7b33b4bdf67dde29cd2773288f1a466f76d928b75de7eb7d9a301d611d16b` |
| `source/casmi26-v44-pairtail-locked-top1-v355245043.py` | 由 Notebook 机械导出的可检索 Python | `42d395c311fc4588c999f700c5f3ae00adea4f033b5a0ac3c4a96f13c0fdf348` |
| `source/notebook_metadata.json` | 原 Notebook 的 Kaggle、环境、输入版本与 Papermill 元数据 | `1d554eee9d59192365140dc6a19f7a4ebade73ef5356265efbe66167c80dd548` |

已验证的最终 `submission.csv`：400 行、400 个唯一 molecule_id、每题 3–25 个候选；SHA256：
`743fcd271cb15de56121da2d58b002b94da450b9c7c0eacba2da72166e89e20d`。
该 CSV 本体仍位于 Kaggle 版本输出；本目录记录其校验值而不把 500 个运行输出全部复制进仓库。

Papermill 元数据记录：开始 `2026-10-04T18:29:15.523937Z`，结束 `2026-10-04T20:31:52.064157Z`，duration `7356.54022 s`，exception 为 null。页面总计时为 7370.8 秒，两者统计边界略有不同。

## 实际执行摘要

| 阶段 | 实际观测 |
|---|---:|
| 主实验谱库 | 2,539,608 spectra / 275,810 structures / 133,228,108 peaks |
| 自建联合池 | 774,943 structures |
| v4 主池 | 710,701 structures |
| 自建 engine 完成 | 400 molecules；约 2,724 s |
| PubChem proposal | 400/400；约 2,674 s；最终 gate 400/400 untouched |
| 主 v4 排名 | 400/400，errors 0；累计约 6,349 s |
| ICEBERG | 71/400 molecules；4,361/26,155 candidates；302.13 s；Top1 改变 0 |
| GLACIER | 仅 `[M+H]+`；366/366；24,003 candidates；524.91 s；Top1 改变 0 |
| RRF + forward rerank | Top1 改变 6；累计约 7,242 s |
| V29 固定 Top1 回填 | 400 个 ID 映射存在；实际回填改变 1 题 |
| 最终验证 | 400 molecules passed |

## 逻辑结构

1. 两套检索/排序引擎并行：v4 160-feature LightGBM（4 boosters）与自建 two-ranker engine。
2. AFIX 用 molecular formula 修正参考谱中性质量；参考代表加入 timsTOF 偏好。
3. 候选池联合 train、COCONUT、ChEBI、LIPID MAPS；使用完整训练折的 `fpnet_full1`。
4. popularity prior 以 `POOLPOP_MU=0.15` 参与池内重排。
5. ICEBERG 与 GLACIER 做 forward rerank；GLACIER 只处理 `[M+H]+`。
6. 两个引擎以 RRF（`ALPHA=0.6`, `KRR=3.0`）融合，再做 forward rerank。
7. 最后使用 400 个可见测试 molecule_id 的 V29 Top1 字典强制保留旧冠军 Top1。

## 重要边界

`0.417` 已被精确复现，但它不是纯粹的可泛化模型成绩。Notebook 最终含固定测试 ID 到 Top1 的映射。下一轮研究可以把此版本作为“排行榜回归基线”和工程参考，但确认性验证与未来比赛推理必须禁用该映射，并通过 ID 重命名不变性测试。

原始输入模型、候选池和外部 Kaggle datasets 没有复制进此仓库；Notebook 会按 Kaggle 挂载路径寻找它们。完整依赖名称和运行路径保留在 `.ipynb`/`.py` 中。

归档验证：Notebook 可由 Jupyter 正常读取；JSON 元数据可解析；NP-PairTail 相关定向回归共 17 项通过。此次只记录和验证，没有再次提交比赛。
