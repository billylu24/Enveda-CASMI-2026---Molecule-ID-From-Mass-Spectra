# CASMI26 项目完整交接与下一轮起点

日期：2026-10-05（America/Los_Angeles）  
当前最高已验证公榜：**0.417**  
基线版本：Kaggle script version `355245043`，完整源码已归档。

## 1. 当前结论

本账户已成功复现公开 V44 PairTail Locked Top1，Kaggle 状态 Succeeded，Public MRR@25 0.417，运行 7370.8 秒（页面 2h 2m 51s，T4 x2）。这是下一轮的排行榜回归基线。

但该版本最后包含 400 个固定测试 ID 的 V29 Top1 字典。它在最终 RRF 后实际改回 1 个 Top1，因此 0.417 不能被解释为纯可泛化模型的独立成绩。下一轮必须同时维护两条线：

- `LB regression`：完整复现 0.417，验证工程没有退化。
- `honest research`：移除固定 ID 映射，在身份隔离的 development/acceptance 上做可泛化改进。

## 2. 代码与资产地图

| 路径 | 角色 | 当前状态 |
|---|---|---|
| `kaggle_release_v44_reproduction/` | 0.417 复现源码、证据和结果清单 | 已冻结 |
| `kaggle_release_np_pairtail/` | 下一轮 NP-PairTail 数据、FPNet 包装和恢复状态 | M0 完成；M1 恢复；M2 未训练 |
| `casmi_ml/np_pairtail_foundation.py` | 身份拆分、AFIX、候选与基础研究流水线 | 已实现并有测试 |
| `casmi_ml/np_pairtail_fpnet.py` | 6 层 Transformer FPNet、三 seed、门槛 | 已实现，未正式训练 |
| `kaggle_release_np_pairtail/build_foundation.py` | Foundation Notebook 生成器 | 已运行过；原作业取消后缓存修复 |
| `kaggle_release_np_pairtail/build_fpnet.py` | FPNet Notebook 生成器 | 已生成，未正式 GPU 运行 |
| `tests/test_np_pairtail_*.py` | foundation、缓存、包装、FPNet 回归 | 已有本地检查 |
| `external/lotus/` | LOTUS v11 元数据、结构投影、许可记录 | 导入完成；CC BY 4.0 |
| `external/chebi/`, `external/lipidmaps/` | 化学候选源 | 已导入 |
| `external/msfinder/` | product ion / neutral loss 字典 | 已转换 |
| `artifacts/` | 历史训练、消融、选择和统计结果 | 保留，不作为新验收 |
| `docs/` | 全部研究报告、PRD、失败复盘 | 完整索引见文件清单 |

全项目非虚拟环境文件清单在 `docs/PROJECT_FILE_INVENTORY_20261005.tsv`，记录 465 个文件及字节数。仓库总占用约 2.4 GB；其中 `external/` 约 777 MB，`artifacts/` 约 16 MB。

## 3. 0.417 流水线的可复用部分

| 组件 | 参数/观测 | 下一轮处理 |
|---|---|---|
| AFIX | formula mass 优先；参考库 2.54M spectra | 保留；在新 acceptance 上做 10/20/30 ppm 消融 |
| 双引擎 | v4 160 特征 4 boosters + two-ranker engine | 保留为两个独立 rank source |
| FPNet | `fpnet_full1`，完整训练折 | 作为教师/回归对照；训练自有三 seed 模型 |
| NP pool | train + COCONUT + ChEBI + LIPID MAPS，774,943 structures | 增加 LOTUS，保留来源特征和许可 |
| popularity | `POOLPOP_MU=0.15` | 只比较 0/0.15/0.30；不得按公榜选择 |
| PubChem | 2674 s，最终 400/400 untouched | 默认关闭；先做 100 题止损试点 |
| ICEBERG | 71/400，302 s，Top1 变化 0 | 改为低置信/高价值题预算路由 |
| GLACIER | `[M+H]+` 366/366，525 s，Top1 变化 0 | 保留加合物域门控，重点测试 tail gain |
| RRF | `ALPHA=0.6`, `KRR=3.0`；6 个 Top1 改变 | 在 dev 冻结参数后一次 acceptance |
| 固定 Top1 | 400-ID 字典，实际回填 1 题 | 仅 LB regression；honest research 禁用 |

## 4. 当前研究分支真实状态

- Git HEAD：`e9e590753f04feeeb8e697e3bfab1c8abd12bff6`。
- 分支：`codex/chemical-priors-and-gan-analysis`。
- Foundation 原运行在 2082.0 秒主动取消，原因是重复计算；不是模型失败。
- 缓存恢复第一版 29.1 秒失败，原因是两个 `train.parquet` 路径歧义；路径解析已修复。
- 累计已消耗作业时间 2111.1 秒，继续计入原 14,400 秒预算。
- M0：119,977 train identities、1,930 dev、2,870 sealed acceptance。
- M1 扫描：1,331,758 reference spectra 保留；52,801 行旧 ppm 超 30 而公式重算回到范围。
- M2 FPNet：代码和 Notebook 已准备，尚未正式训练，不得写成已获得提升。
- 当前工作树原有未提交内容：NP-PairTail FPNet 文件与两份状态/PRD修改；本次新增 0.417 归档和交接文档。不要用 reset/checkout 覆盖。

## 5. 下一次运行：量化执行顺序

### Gate A — 归档与回归（≤15 分钟人工检查）

1. 校验 V44 Notebook SHA256 与 `verified_result.json`。
2. 确认导出的 Python 包含 21 个 code cells 的内容和固定 Top1 映射。
3. 对未来任何打包版本运行 ID 重命名不变性测试；honest 分支必须逐题候选完全不变。

通过标准：哈希一致；0 个缺失文件；honest 分支中 0 个 `m_... → prediction` 映射。

### Gate B — 恢复 M1（剩余总预算 12,288.9 秒）

用缓存版 Foundation 继续 C1–C4，禁止改变拆分盐、质量窗口、候选池或评价规则。

通过标准：

- C0 缓存/非缓存逐题排名完全一致。
- 1,930 个 development identities 全部完成 4 个候选臂。
- 无 NaN/Inf；失败题为 0，或每题明确记录原因且比例 <0.5%。
- 保存候选 recall@25、MRR@25、Top1、每题耗时和来源漏斗。

### Gate C — M2 FPNet（三 seed，目标 ≤135 分钟 GPU）

- 结构：6 layers、hidden 512、8 heads、dropout 0.1。
- seeds：20261004/20261005/20261006。
- 有效 batch 64；最多 20 epochs；最少 6；patience 4。
- checkpoint 只按 development BCE 选择。

通过标准：三 seed 均无 NaN/Inf；ensemble dev BCE 相对位频率常数基线改善 ≥10%；candidate-truth-present 的 unknown dev MRR 对随机排序 10,000 次 bootstrap 下界 >0。

### Gate D — 单因素消融

固定 B0 后只运行以下增量，禁止一次同时改多个旋钮：

| 实验 | 唯一变化 | 主要指标 |
|---|---|---|
| E1 | AFIX on/off | unknown/known MRR、candidate recall |
| E2 | LOTUS source on/off | truth recall@25、source funnel |
| E3 | own FPNet vs `fpnet_full1` | BCE、unknown MRR、校准 |
| E4 | GLACIER on/off，仅 `[M+H]+` | tail MRR、Top1 regressions、coverage |
| E5 | ICEBERG budget router on/off | gain per GPU-second |
| E6 | popularity μ 0/0.15/0.30 | dev MRR 与来源偏差 |
| E7 | RRF 参数固定组合 | rank 2–25 gain 与 Top1 harm |

每项至少报告：ΔMRR@25、ΔTop1、Recall@25、changed rows、paired bootstrap 95% CI、wall time、GPU time。

### Gate E — 冻结与一次 acceptance

先在 development 选择唯一候选，再解封 2,870 个 acceptance identities，一次运行。

发布门槛：

- unknown ΔMRR@25 > 0，95% CI 下界 ≥0。
- known ΔMRR@25 ≥ -0.001，且 95% CI 下界 ≥-0.003。
- Top1 保护不能读取真值、测试 ID 或旧 submission。
- 动态 ID、1–25 合法候选、官方身份去重、400/400 行验证通过。
- 总 Kaggle 推理预估 <8 小时，目标 <3 小时。

若不通过：保留完整阴性结果，不提交比赛，不通过公榜反复调参。

## 6. 最优先的工程优化

1. 删除 400/400 untouched 的全量 PubChem 默认计算，预计节约约 2,674 秒。
2. ICEBERG 从全题候选构造改为低置信/高预期收益路由；以 `ΔMRR/GPU-second` 决定预算。
3. GLACIER 保持 `[M+H]+` 域过滤，但不默认保护 Top1；在独立集决定只排 tail 还是允许 Top1。
4. 所有候选保留来源位和首次进入阶段，定位收益来自召回还是排序。
5. 将 LB regression 与 honest research 生成两个独立 submission 文件，文件名和 manifest 必须注明是否含锁定映射。

## 7. 下一位执行者的第一步

先读：

1. `kaggle_release_v44_reproduction/README.md`
2. `docs/PRD_FORWARD_NP_PAIRTAIL_RUN_20261004.md`
3. `docs/NP_PAIRTAIL_RUN_20261004.md`
4. `kaggle_release_np_pairtail/run_status.json`
5. `kaggle_release_np_pairtail/protocol_fpnet_20261004.json`

然后只恢复缓存版 Foundation，不先改模型参数。M1 完成且报告通过后才启动 M2。

## 8. 本次归档验证

- 下载的 Notebook 可被 Jupyter 正常解析并重新序列化。
- `verified_result.json` 通过 JSON 语法检查。
- 0.417 归档的三个源文件 SHA256 已写入 `kaggle_release_v44_reproduction/SHA256SUMS`。
- NP-PairTail 定向回归：`17 passed in 2.09s`（运行时显式设置 `PYTHONPATH=.`）。
- 未自动提交 Git commit，未覆盖原工作树修改，未重新提交 Kaggle。
