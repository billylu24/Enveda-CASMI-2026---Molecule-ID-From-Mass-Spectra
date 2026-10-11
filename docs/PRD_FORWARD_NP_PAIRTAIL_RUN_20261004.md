# CASMI26 下一次训练与运行 PRD：NP-PairTail Forward Cascade

版本：1.0  
日期：2026-10-04（America/Los_Angeles）  
状态：待实施；本文定义下一次研究训练、独立验收和发布门槛，不代表模型已经训练或成绩已经获得。

关联文档：

- [V1 优化实际结果](V1_OPTIMIZATION_RESULTS_20261003.md)
- [下一轮通用研究规格](PRD_NEXT_RUN_20261003.md)
- [class 3 与结构生成路线](DE_NOVO_NEXT_STEPS_20261002.md)
- [化学先验与公共结构库](CHEMICAL_PRIORS_20261001.md)
- [固定测试 ID 风险](LEADERBOARD_ID_CAVEAT_20261003.md)

## 1. 决策摘要

下一次不继续提高 V1 的单一保护阈值，也不把 GAN 或 CycleGAN 作为主线。主线改为：

1. 用分子式精确质量修正参考谱索引（AFIX），并使参考谱代表选择感知极性、加合物和仪器。
2. 在严格身份隔离的数据上训练谱图到结构指纹模型（FPNet）和候选排序器。
3. 将 COCONUT、LOTUS、ChEBI、LIPID MAPS 等候选保留来源标签后联合，而不是无差别混库。
4. 对候选结构使用冻结的结构到谱图模型进行 forward rerank；模型只在其训练域内启用。
5. 融合独立排序引擎时使用 reciprocal-rank fusion（RRF），避免直接平均未校准分数。
6. 冻结可信 Top1 的条件保护规则，重点改善第 2—25 名；但禁止按可见测试 `molecule_id` 查表锁定旧答案。

确认性主问题是：

> 在不使用固定测试 ID、不查看独立验收标签且不损害 known 检索的前提下，NP 候选、训练后的 FPNet、前向谱图分数和 RRF 能否稳定提高 unknown MRR@25？

## 2. 已知证据与问题定义

### 2.1 当前基线

冻结基线 B0 为 V1：历史公榜 0.176，原正式运行 417 秒。本轮独立研究重放了相同逻辑，并得到：

| 队列 | 分子数 | B0 MRR@25 | B0 Top1 |
|---|---:|---:|---:|
| development unknown | 481 | 0.016706 | 需从保存结果重算 |
| development known | 426 | 0.782842 | 需从保存结果重算 |
| acceptance unknown | 970 | 0.015276 | 0 |
| acceptance known | 832 | 0.781046 | 需从保存结果重算 |

提高保护阈值到 0.95 的验收结果未通过：unknown 差值仅 +0.000340，95% 配对区间为
[-0.001236, +0.002081]；known 差值区间下界为 -0.001073，低于预设保护线。因此下一轮不能继续把“打开更多扩展候选”当作充分改进。

### 2.2 三条外部研究线索

1. Enveda 的研究重点指向天然产物和数据库外化学空间。COCONUT、LOTUS、NPAtlas 等资源更适合作为有来源先验的候选库，而不是只扩大通用 PubChem 池。
2. 结构到谱图的前向模型可以验证检索或生成候选是否真正解释实验谱图。GrAFF-MS、ICEBERG、GLACIER 等均支持该方向，但必须遵守加合物、仪器和碰撞能量训练域。
3. 已核查的公开 0.417 Notebook 采用 AFIX、timsTOF 偏好、FPNet、160 特征 LightGBM、ICEBERG/GLACIER、RRF 和 popularity prior；最终还按 400 个固定测试 ID 强制恢复旧版 Top1。后者不可作为可泛化方法。

### 2.3 从 0.417 Notebook 得到的可量化事实

- 运行时间约 2 小时 15 分钟。
- 主实验谱库约 253.96 万张谱、27.58 万个结构。
- 主候选池约 71.07 万结构；自建引擎候选池约 77.49 万结构。
- PubChem-only 阶段运行约 2957 秒，但最终门控统计为 400/400 `untouched`，没有候选进入提交。
- ICEBERG 在约 300 秒预算内只完成 71/400 题、4361/26156 个候选。
- GLACIER 仅对含 `[M+H]+` 的题目运行，完成 366/366 题、24004 个候选，约 591 秒。
- ICEBERG 和 GLACIER 改变了大量 Top25 顺序，但日志中各自改变 Top1 为 0 题。
- 双引擎 RRF 后有 6 题 Top1 变化；固定旧 Top1 映射最终回滚其中 1 题。

这些事实说明下一次应优先研究尾部排序、前向覆盖率和候选召回，不应把完整 PubChem 扫描或固定 Top1 映射作为默认路径。

## 3. 产品目标与非目标

### 3.1 必须交付

1. 一个可恢复的研究入口：准备数据、训练 FPNet、训练 ranker、开发选择、冻结、一次独立验收。
2. 一个只加载冻结资产的比赛推理入口：不训练、不运行研究队列、不读取旧测试答案。
3. AFIX、参考谱代表选择、NP 候选、FPNet、forward rerank、RRF 的逐项消融。
4. 每题候选漏斗、来源、前向覆盖、排名变化、耗时和失败原因。
5. 固定协议下唯一候选相对 B0 的配对统计结论；允许结论为“不通过”。
6. 若通过，生成动态测试 ID 的合法提交并记录真实 Kaggle 状态和分数。

### 3.2 本轮明确不做

- 不实现或训练完整 CycleGAN；公开 dMSdMol 项目没有已完成的 CycleGAN 结果。
- 不把指纹 GAN 当作默认模型；已有 matched 对照没有稳定收益。
- 不在本轮训练通用 SMILES/分子图解码器；class 3 生成保留为后续独立项目。
- 不按公开榜单逐次调整阈值、权重或 400 个测试答案。
- 不承诺 0.417 或任何特定公榜成绩；该分数包含复杂集成和固定 ID 混杂因素。
- 不因候选库变大就宣称能力提升；必须以真值召回和最终排名衡量。

## 4. 系统数据流

```text
实验谱图组
  ├─ 精确中性质量、加合物、模式、仪器、碰撞能量
  ├─ 实验谱库检索与 shifted analog
  ├─ 训练后的 spectrum→fingerprint FPNet
  └─ 多谱一致性/元数据质量
             ↓
候选联合池
  ├─ train/library structures
  ├─ COCONUT
  ├─ LOTUS
  ├─ ChEBI / LIPID MAPS
  ├─ 许可通过后才允许 NPAtlas
  └─ 小规模门控 PubChem proposal（默认关闭）
             ↓
基础候选特征与 LightGBM ranker
             ↓
结构→谱图 forward scoring
  ├─ GLACIER：仅兼容加合物
  ├─ ICEBERG：仅在预算覆盖可保证时启用
  └─ GrAFF-MS：仅兼容模型训练域的诊断臂
             ↓
RRF 融合独立引擎
             ↓
条件 Top1 保护 + PairTail 重排
             ↓
官方身份去重、Top25、动态 submission.csv
```

## 5. 数据、身份隔离与防泄漏

### FR-01：身份单位

训练、开发、验收、候选去重和评价统一采用比赛指定的互变异构规范化后 InChIKey 前 14 位。保存原始 SMILES、规范 SMILES、原始 InChIKey、规范身份和别名映射。

任何同身份别名不得跨训练、开发和验收。谱图副本、分箱后完全相同谱和同一原始记录不得跨集合。

### FR-02：拆分

旧 development/acceptance 和已经查看过的失败案例只作工程回归，不再作为新的确认性验收。下一次训练必须构造新拆分：

- 拆分盐：`casmi26-np-pairtail-20261004-v1`。
- 身份级比例：训练 80%、开发 10%、验收 10%。
- 每个身份训练最多 8 张谱；开发和验收最多 4 张谱，保留多仪器/多能量多样性。
- 若训练身份超过 120,000，按固定身份哈希裁剪到 120,000；不得按结果挑选。
- 开发目标至少 1,000 个身份；验收目标至少 1,500 个身份。实际不足时报告样本量并将统计结论标为证据不足。
- 所有已恢复的历史开发、验收和手工检查身份从新验收排除。

另建立一个 scaffold-distance 诊断切片，但除非训练身份按 scaffold 完全隔离，不得称为严格 scaffold split。

### FR-03：known 与 unknown

- `known`：删除查询谱本身及数值副本，允许保留同身份的其他合法实验参考谱。
- `unknown`：从实验参考谱库移除查询身份及全部别名；公共结构候选库仍按真实部署规则保留。
- `database-absent diagnostic`：额外从结构候选池移除查询身份，只用于诊断生成上限，不参与本轮主验收。

真值只能进入评估器。候选生成、门控、路由、Top1 保护和停止条件均不得读取真值存在性或最终排名。

### FR-04：固定 ID 禁令

代码、模型、配置和发布包不得包含：

- `molecule_id → SMILES/InChIKey/rank` 映射；
- 旧 submission 行；
- 针对 `m_...` 的条件分支；
- 可见测试 ID 哈希或顺序表。

必须通过任意 ID 重映射测试：相同谱图和元数据随机改名后，再将输出 ID 恢复，候选身份和顺序必须逐题完全一致。

## 6. 候选层规格

### FR-05：AFIX 参考质量

参考谱索引优先使用其已知结构 `molecular_formula` 计算中性单同位素质量。公式缺失、解析失败或包含未知元素时才回退到前体与加合物反推质量。

每条参考记录保存：

- `mass_source ∈ {formula, precursor_adduct, unavailable}`；
- 公式质量与前体反推质量差值 ppm；
- 加合物解析状态；
- 是否因差值异常被降权或排除。

开发前固定异常阈值候选为 10、20、30 ppm，最多比较三种；验收前选定唯一阈值。不得直接扩大查询质量窗来掩盖参考质量错误。

### FR-06：参考谱代表

同一结构的代表谱按以下稳定顺序选择：

1. 查询极性相同；
2. 加合物相同，其次为预声明的兼容加合物；
3. 仪器家族相同；
4. 若查询为 timsTOF，优先 timsTOF；其他仪器不得无条件偏好 timsTOF；
5. 有效峰更多；
6. 稳定记录 ID。

对照必须包含“无仪器偏好”和“上述仪器感知”两臂。

### FR-07：天然产物候选池

各来源保留独立位标志和版本：

- `train/library`
- `COCONUT`
- `LOTUS`
- `ChEBI`
- `LIPID_MAPS`
- `NPAtlas`（仅许可审查通过后）

联合时按官方身份去重，保留全部来源集合，不因来源合并覆盖原 SMILES 或质量。每题记录各来源进入质量窗、进入初始候选、进入 Top25 和首次命中真值的数量。

NP 来源只作软特征，不作硬过滤。必须单独报告天然产物来源切片与非天然产物切片，避免总体收益掩盖另一侧退化。

### FR-08：PubChem proposal 止损

全量 PubChem 通道默认关闭。先在开发集中固定选择 100 题试点，其中至少 50 题为 B0 低置信度：

- 记录质量窗结构数、两阶段指纹筛选时间、非现有池候选数和实际进入 Top25 数。
- 若进入最终 Top25 的题少于 5%，或新增真值 Recall@25 小于 0.5 个百分点，终止全量 PubChem 阶段。
- 若每题中位耗时超过 20 秒，除非有已验证的索引优化，否则终止。
- 终止必须保存报告，不得像已研究 Notebook 一样支付近 3000 秒后再由门控拒绝全部候选。

## 7. FPNet 训练规格

### FR-09：任务

输入一个分子的单谱或多谱集合，输出候选池一致定义的结构指纹 logits。第一轮使用固定选定位集合；位集合只能由训练身份拟合。

输入特征：

- 峰 m/z 多尺度正弦嵌入；
- `precursor_mz - fragment_mz` 中性丢失嵌入；
- 平方根归一化强度；
- 前体 m/z；
- 加合物、极性、仪器家族；
- 碰撞能量均值、最小值、最大值、缺失标志；
- 单谱/合并谱标志及合并谱数。

碰撞能量保持原始 `eV` 定义；不得把 eV 静默当作 normalized collision energy。使用要求 NCE 的外部模型时，若不存在可靠换算，则能量设为 unknown 并单独报告。

### FR-10：模型与训练

- 主模型：6 层谱峰 Transformer，隐藏维 512，8 heads，dropout 0.1。
- 两个视图：`single` 对每张谱预测后按分子平均；`merged` 按极性合并多谱后预测。
- 融合：开发前固定为 `0.5 × single + 0.5 × merged`，另仅允许比较 `single-only` 作为消融。
- 损失：带每位正样本权重上限的 BCEWithLogits；权重只从训练集统计。
- 优化器：AdamW，初始学习率 1e-4，weight decay 1e-4，2 epoch warmup，随后 cosine decay。
- batch size：按显存选择 32/64，但有效 batch 必须通过梯度累积统一为 64。
- 最多 20 epochs；最少 6 epochs。
- early stopping：开发 fingerprint BCE 连续 4 epochs 无改善。
- checkpoint 选择：最低 development BCE；不得按候选 MRR 在验收集选择。
- 随机种子：`[20261004, 20261005, 20261006]`；开发选择后使用三个 seed 的 logits 等权平均。
- 每个 seed 最长训练 45 分钟；三 seed 总训练目标不超过 135 分钟。

每 epoch 记录 train/dev BCE、micro/macro AUROC、bit prevalence calibration、学习率、实际 batch、GPU 时间和峰值显存。AUROC 对单类位标记为不可计算，不填 0 或 1。

### FR-11：FPNet 成功门槛

以下只决定模型是否可以进入 ranker 训练，不直接决定最终发布：

- 三 seed 均完成至少 6 epochs，无 NaN/Inf。
- ensemble development BCE 优于训练集位频率常数基线至少 10%。
- 在候选真值已存在的 development unknown 子集上，FPNet 单独排序的 MRR@25 高于随机同窗候选排序；10,000 次配对 bootstrap 下界大于 0。
- 单谱和 merged 融合相对 single-only 的 development MRR 不得下降超过 0.001，否则使用 single-only。

## 8. 候选排序器训练

### FR-12：特征组

候选排序器只使用运行时可获得特征：

1. 实验谱库：直接熵/余弦相似度、质量误差、谱数和多谱一致性。
2. analog：shifted similarity、近邻权重、候选与近邻 Tanimoto。
3. FPNet：raw dot product、长度归一分数、候选内 z-score、rank percentile。
4. 碎裂近似：一/二键断裂可解释峰强度比例及中性丢失覆盖。
5. 元数据：模式、加合物、仪器、碰撞能量缺失、谱图数量、前体质量段。
6. 来源：候选来自哪些结构库、来源数和 NP 来源标志。
7. 化学先验：仅经审核规则的正向软证据。
8. 交叉特征：谱库与 FPNet 一致性、analog 与 FPNet 一致性、来源与质量误差。

禁止特征：molecule_id、测试顺序、真值是否在池、B0 是否命中、公开榜单反馈、旧提交排名查表。

### FR-13：模型

- 模型：LightGBM LambdaRank 或等价组内 ranking objective；组为查询分子。
- 若现有特征资产只支持二分类，必须额外输出组内排名指标，不以全局 AUC 替代。
- 4 个 booster，种子 `[20261004, 20261005, 20261006, 20261007]`，最终分数平均。
- 最大 800 trees，learning rate 0.03，max depth 候选 `{5, 6}`，min data in leaf 候选 `{50, 100}`。
- 开发网格最多 4 个组合；先满足 known 保护，再最大化 unknown MRR，平局选择更浅模型。
- 训练负例必须来自训练身份的真实质量窗与扩展池；每题最多 256 个负例，包含高谱图相似、高 FPNet 分数和随机候选。

## 9. Forward cascade 与 PairTail

### FR-14：训练域门控

每个 forward model 必须有机器可读适用范围：支持的加合物、极性、仪器、碰撞能量形式、质量范围和最大原子数。

- GLACIER：仅对其权重明确支持的 `[M+H]+` 谱图启用。
- GrAFF-MS：默认只作诊断；仅在 `[M+H]+/[M-H]-`、兼容仪器且碰撞能量处理被明确验证时进入融合。
- ICEBERG：只有全量预算能覆盖至少 80% 的 eligible 分子时才可进入确认性候选；否则保留为开发诊断。

模型域外返回 missing，不得返回零分冒充负证据。

### FR-15：前向分数

每个候选至少计算：

- predicted/observed cosine；
- modified cosine 或等价质量容差匹配；
- 预测峰对观测强度的覆盖率；
- 观测峰被预测解释的覆盖率；
- 多张谱的 max、mean、top-3 mean；
- 不同 forward 模型间的一致性。

所有分数在每题候选内使用训练/开发冻结的稳健标准化。候选数小于 3 或标准差近零时使用确定性退化规则，不产生 NaN。

### FR-16：PairTail 与 Top1 保护

PairTail 的定义是：允许新证据重排 Top2—Top25，并只在满足预先训练的置信条件时挑战 B0 Top1。

Top1 保护模型只允许使用：

- B0 top1 分数和 top1/top2 margin；
- 同身份多谱一致性；
- 质量误差与加合物一致性；
- FPNet、forward 和两个独立排序器是否一致；
- 模型是否处于训练域。

开发时比较不超过三个保护策略：

1. `tail_only`：永不改变 B0 Top1；
2. `consensus_unlock`：至少两个独立新证据同时反对 B0 Top1，且候选分差超过开发阈值；
3. `free_rerank`：不保护 Top1，仅作上限诊断。

验收前只选择一个。无论选择哪一个，都不得使用固定 ID 或旧答案映射。

### FR-17：RRF

主 ranker 与独立 secondary engine 使用：

\[
S(c)=\frac{1}{K+r_1(c)}+\frac{\alpha}{K+r_2(c)}
\]

开发候选限定为：

- `K ∈ {3, 10}`；
- `alpha ∈ {0.3, 0.6}`。

最多四个组合。缺席某引擎的候选只获得已出现引擎的项，不填虚构末位。相同身份先合并来源再排序。

### FR-18：Popularity prior

Popularity 只对公共数据库检索候选生效，不对生成候选施加零分惩罚。开发权重只允许 `{0, 0.05, 0.10, 0.15}`。

它是次要消融，不进入默认确认性系统。仅当 development unknown MRR 提升至少 0.001、known 不下降超过 0.001、天然产物低频切片不下降超过 0.002 时，才允许进入冻结候选。

## 10. 冻结实验矩阵

| 臂 | 相对前一臂的唯一变化 | 角色 |
|---|---|---|
| C0 | 冻结 V1 B0，保留身份去重 | 主基线 |
| C1 | C0 + AFIX | 参考质量贡献 |
| C2 | C1 + 仪器/加合物感知代表谱 | 参考选择贡献 |
| C3 | C2 + 新 FPNet 与新 LightGBM ranker | 学习排序贡献 |
| C4 | C3 + 带来源标签的 NP 候选池 | 天然产物候选贡献 |
| C5 | C4 + 训练域门控的 forward cascade | 前向重排贡献 |
| C6 | C5 + secondary engine RRF + PairTail | 唯一确认性候选来源 |
| C7 | C6 + popularity prior | 次要探索，不默认发布 |
| D1 | C6 + 受控 PubChem proposal | 仅通过 FR-08 pilot 后运行 |

开发阶段依次计算 C0—C6。每一步必须同时保存前一步结果；若某一步不满足以下局部门槛，则默认回退到前一步继续后续组合：

- unknown development MRR 不下降超过 0.0005；
- known development MRR 不下降超过 0.001；
- known Top1 不下降超过 0.002；
- 运行时间增量不超过该模块预算；
- 没有身份、格式或覆盖错误。

最终只能冻结一个确认性候选，与 C0 在新验收集上比较。C7、D1 即使点估计更高，也不能在看验收后替换正式候选。

## 11. 指标与验收门槛

### 11.1 主指标

- 主比较：冻结候选 − C0 的 unknown MRR@25。
- 配对单位：分子身份。
- bootstrap：10,000 次，种子 20261004。
- 通过条件：差值 ≥ +0.001，且 95% 配对区间下界 > 0。

### 11.2 known 保护

- known MRR 差值 95% 区间下界 ≥ −0.001。
- known Top1 点估计下降 ≤ 0.002。
- known Recall@25 点估计下降 ≤ 0.002。

三项必须同时通过。

### 11.3 Tail 指标

以 C0 真值初始排名固定分层，不随新模型改变分母：

- `tail_reciprocal_rank`：仅统计 C0 排名 2—25 的题。
- `rescued@25`：C0 在 Top25 外、新模型进入 Top25 的题数。
- `lost@25`：C0 在 Top25 内、新模型掉出 Top25 的题数。
- `promoted_to_top1` 和 `demoted_from_top1`。
- `net_tail_gain = Σ(RR_new − RR_C0)`。

PairTail 进入确认性候选的开发门槛：`net_tail_gain > 0`，`rescued@25 > lost@25`，且 known Top1 保护满足。

### 11.4 候选指标

每个来源及联合池报告：

- 质量窗 Recall；
- candidate Recall@25/50/100；
- 新增真值数；
- 原真值因竞争被挤出数；
- 每题候选中位数、P90、最大值；
- 身份重复率和非法 SMILES 率。

“库中存在真值”与“最终排进 Top25”必须分开。

### 11.5 Forward 指标

- eligible 分子数；
- 成功评分分子数与覆盖率；
- 成功候选数与失败候选数；
- 每秒候选数；
- forward 加入前后 MRR、Top1、Tail 指标；
- 域内与域外切片；
- 按加合物、仪器、质量和天然产物来源分层。

确认性 forward cascade 要求 eligible 分子覆盖率 ≥ 90%；其中任一参与融合的模型覆盖率低于 80% 时，该模型只能作为 missing 特征或探索臂，不能以不完整运行结果代表全系统。

## 12. 资源与时间预算

| 阶段 | 目标预算 | 硬停止/降级规则 |
|---|---:|---|
| 数据/身份审计 | 20 分钟 | 交集或 manifest 错误立即停止 |
| AFIX 与参考索引构建 | 25 分钟 | 超时先缓存优化，不扩大规模 |
| FPNet 单 seed | 45 分钟 | 保存最后完整 epoch；不得使用半写 checkpoint |
| 三 seed FPNet | 135 分钟 | 任一 seed 失败则 ensemble 不进入确认性实验 |
| Ranker 特征与训练 | 45 分钟 | 超时减少开发网格，不减少验收数据 |
| 100 题 forward pilot | 20 分钟 | 估算全量覆盖与吞吐，决定预算 |
| 全量 forward development | 60 分钟 | 低覆盖模型退出确认性融合 |
| 一次新 acceptance | 60 分钟 | 不得中途看结果调权重 |
| 研究全流程 | 4 小时 | 超预算记为工程不通过，不静默截断候选 |
| 正式可见 400 题推理 | 目标 ≤ 150 分钟 | >180 分钟不得发布；9 小时限制保留充足余量 |

内存门槛：CPU 峰值目标 ≤ 24 GB，GPU 峰值目标 ≤ 14 GB/卡。实际 Kaggle 机器规格不同则记录真实上限；不得以 OOM 后静默减少候选数作为同一配置继续运行。

进度要求：超过 60 秒的阶段每 60 秒或每 25 个分子写一次状态；内容包括完成数、总数、速度、累计错误、显存/内存和按当前速度估算的剩余时间。

## 13. 运行阶段与停止条件

### M0：协议冻结

产物：`protocol_np_pairtail_20261004.json`、身份列表、已观察身份列表、数据许可清单、模型适用域清单。

通过条件：

- 训练/开发/验收身份交集为零；
- 所有输入、代码、规则、模型和索引均有 SHA256；
- NPAtlas 或其他非商业数据的使用资格明确；不明确则移出本轮。

### M1：工程基线与 AFIX

运行 C0/C1/C2，小批量 100 题后再全开发集。

停止条件：

- C0 不能复现冻结 B0；
- AFIX 公式解析错误率 >1%；
- 参考质量异常无法解释或出现跨身份污染；
- ID 重映射测试失败。

### M2：FPNet 训练

运行三个 seed，训练结束后冻结 checkpoint 和指纹位定义。

停止条件：

- NaN/Inf；
- 三 seed 中任一未完成最低 epoch；
- 常数基线改善不足 10%；
- 开发候选排序没有显著优于随机。

### M3：Ranker 与 NP 候选

依次评估 C3、C4；保存逐题完整排名和候选来源。

停止条件：

- 候选池增大但 Recall 不增、MRR 明显下降；
- NP 低频切片退化超过 0.002；
- 训练特征出现测试 ID、验收标签或旧提交信息。

### M4：Forward pilot 与全量开发

先测 100 题吞吐、覆盖和增益，再运行符合预算的模型。选择一个 PairTail 策略和一个 RRF 配置。

停止条件：

- eligible 覆盖不足；
- forward 失败被错误转成零分；
- Top1 下降超过保护线；
- PubChem proposal 未通过 FR-08。

### M5：冻结与一次验收

写入唯一候选的全部配置和 SHA，再解封新 acceptance。只比较冻结候选与 C0。

通过条件：第 11 节所有主指标和 known 保护同时通过。未通过时保存负结果，禁止在同一验收集改参数后重试。

### M6：发布

比赛 Notebook 只加载冻结资产并处理实际 test。不得包含训练集划分、开发/验收评价、固定 ID 字典或旧 submission。

只有以下条件全部满足才提交：

- M5 通过；
- 动态 ID、候选数、合法性和官方身份去重通过；
- 400 题正式推理 ≤180 分钟；
- release manifest 与研究冻结 SHA 完全一致；
- 输出不依赖可见测试 ID。

## 14. 必须保存的产物

| 产物 | 内容 |
|---|---|
| `protocol_np_pairtail_20261004.json` | 拆分、种子、网格、预算、门槛、唯一候选规则 |
| `data_manifest.json` | 数据来源、版本、许可、SHA、结构与谱图数量 |
| `split_manifest.json` | 身份级训练/开发/验收列表和排除原因 |
| `fpnet_manifest.json` | 架构、指纹位、checkpoint、训练曲线与 SHA |
| `ranker_manifest.json` | 特征名、LightGBM 参数、训练身份和 booster SHA |
| `forward_domain.json` | 每个模型的适用加合物、仪器、能量、质量范围 |
| `candidate_funnel.parquet` | 每题每来源的进入、淘汰、首次真值位置 |
| `development_results.json` | C0—C7 指标、局部门槛和选择记录 |
| `acceptance_results.json` | 唯一主比较、bootstrap 和通过/拒绝结论 |
| `runtime_profile.json` | 阶段耗时、吞吐、覆盖、CPU/GPU 内存 |
| `release_manifest.json` | 代码、数据、权重、规则、索引和输出 SHA |
| `run_status.json` | 当前阶段、进度、错误、Kaggle 状态和实际分数 |

逐题完整预测、训练身份和大模型是否可以公开由数据许可决定；即使不提交 Git，也必须保存在可审计运行归档中。

## 15. 必测回归

1. 任意替换全部 query ID，恢复 ID 后预测逐题相同。
2. test 为 37、400、800 题时均按实际 ID 输出，无固定长度假设。
3. 同身份不同 SMILES/互变异构体只保留首个官方身份，不补入重复项。
4. 公式质量存在时 AFIX 不受错误参考前体影响；公式失败时确定性回退。
5. 查询没有兼容加合物时 GLACIER 不运行，且 missing 不等于零分。
6. ICEBERG 达到预算时明确报告未覆盖题，不把部分结果描述为全量。
7. Popularity 不惩罚生成候选；关闭权重时顺序与前臂完全一致。
8. PubChem gate 0% 通过时在 pilot 后停止，不启动全量扫描。
9. 候选来源合并不覆盖原结构、质量或来源集合。
10. 训练/开发/验收同身份、别名和谱副本检查均为零。
11. CPU/GPU 的数值差异不得无解释地改变最终 Top25 身份顺序。
12. 缓存或模型 SHA 不一致时 fail closed，不静默使用旧资产。

## 16. 完成定义

本轮只有三种合法结论：

1. **接受**：确认性 unknown 与 known 门槛同时通过，工程预算通过，发布并取得真实比赛状态。
2. **拒绝**：模型或组合未通过预设科学门槛，保留 C0，完整记录负结果。
3. **证据不足**：样本量、覆盖、许可、运行失败或资源不足使确认性结论不可成立。

训练完成、loss 下降、候选更多、Notebook 成功、公开榜分数偶然升高，都不能单独定义为本轮成功。成功的最低条件是：身份隔离、动态 ID、一次冻结验收、known 保护、unknown 稳定提升和可复现发布同时成立。

## 17. 原版 0.417 复现运行（基线取证）

本节记录对原作者公开 Version 3（`scriptVersionId=354936340`）的原样复现。它只用于建立可比较的工程基线，不替代本 PRD 的无固定 ID 验收方案。

| 项目 | 当前值 |
|---|---|
| 复现 Notebook | `xiaoyuzhoux120/casmi26-v44-pairtail-locked-top1` |
| Kaggle 版本 | `scriptVersionId=355245043` |
| 版本名 | `Reproduce original V3 354936340` |
| 启动时间 | 2026-10-04 11:29 PDT |
| 资源 | GPU T4 x2 |
| 输入 | 比赛数据 + 14 个数据集，共 15 个输入源 |
| 环境 | 固定到原始 2026-06-30 环境；Internet off |
| 最终状态 | Successfully ran |
| 总运行时 | 7370.8 秒（2 小时 2 分 51 秒） |
| 初始校验 | RDKit 2026.03.3；64 个文件匹配 `MANIFEST.json`；V4/V3、pool、PubChem 与 popularity prior 均发现 |
| 关键阶段 | 主引擎 400/400、0 errors；PubChem 400/400；ICE 71/371 covered（300 秒预算停止）；GLACIER 366/366；最终验证 400 molecules |
| 输出文件 | `submission.csv`，393,908 bytes（页面显示 393.91 kB） |
| 输出校验 | 400 行、400 个唯一 `molecule_id`、每题 3—25 个候选、无空 ID；Notebook 内 RDKit 最终合法性校验通过 |
| SHA-256 | `743fcd271cb15de56121da2d58b002b94da450b9c7c0eacba2da72166e89e20d` |
| 自动提交 | 禁止；仅生成并检查 `submission.csv` |

原版行为已成功复现；但尚未提交比赛，因此不能把 0.417 榜分视为已经独立复核。原版包含 400 个 `molecule_id -> top1` 的 PairTail 锁定字典，因此本结果只视为“原版行为复现”；下一次正式训练必须使用第 6、11、15 节规定的动态 ID 与反查测试。
