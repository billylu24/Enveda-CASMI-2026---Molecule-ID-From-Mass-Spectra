# NP-PairTail 缓存重训：新实验声明

用户在上一轮失败复盘后明确要求“重新训练并提交kaggle”。据此启动独立工程修复实验 `np-cache-retrain-20261005-v1`。此前实际作业累计 10528.1 秒保留；此次新的研究预算为 14400 秒，后续阶段共用，不将旧成本抹掉。旧失败 Version 2 与嵌入源包保持原样，所有新代码与产物放在独立发布目录。

三个种子从头训练，模型、损失、身份拆分、质量窗、预处理、开发门槛、最大/最低轮次与封存验收保持原冻结协议。旧权重不使用，缺少随机状态的旧 checkpoint 不冒充无缝恢复。唯一训练工程变更为 AMP 跳步后的整有效批次 FP32 重试，以及今后完整轮次保存随机状态。

## 执行与证据要求

1. 挂载 FPNet 失败 Version 2（355386045）和成功 Cached Foundation Version 2（355258706），核验冻结源、基础文件 SHA、候选池 SHA。
2. 只读复用 train/development 缓存，校验源 parquet 哈希、身份顺序、所有视图的唯一所有权、维度、有限特征、二值指纹，逐文件保存 SHA。拒绝缓存不匹配；不回退到约 73 分钟冷构建。
3. 先在真实双卡上使用实际网络和 64 个缓存样本，主动触发一次受控梯度溢出，验证跳步不改权重、整批 FP32 重试的确定性及恰好一次优化器更新。预检权重不进入正式训练，三 seed 分别重置随机种子。
4. 运行三种子训练、常数 BCE 门槛和同窗排序门槛。未通过则保留实际失败和日志，不跳过门槛。
5. 通过 M2 后继续 ranker、候选增量和适用模型的开发评估；外部 forward 资格不全时保持禁用。冻结一套配置后，仅一次封存验收。
6. 验收与推理格式通过后实际提交比赛，直到具体记录 Succeeded 和公榜出分再报告新成绩。训练启动或 Notebook 成功均不等于比赛提交成功。

本地 36 项相关测试通过。正式 GPU 预检及重新训练尚未完成；初始发布状态为 prepared_not_executed。

## 实际 Version 1 启动失败（355524627）

已正式保存 GPU T4 x2 / Internet OFF，页面代码逐字核验 67581 字符与冻结源一致。Kaggle 在运行代码前报告 `ERRORED_MOUNTING_DATASET`：失败 FPNet Version 2 输出经 30 次挂载重试仍失败。未执行缓存核验、GPU 预检或任何 seed。此记录不归因于模型，也不计为已完成训练。正在通过 Kaggle Output 的 New dataset 功能尝试服务器端转存缓存，避免本地传输及冷重建。页面未提供这次失败的精确 worker 耗时，不能杜撰秒数。

## 缓存转存成功

通过 Kaggle Output → New dataset 服务器端转存，私有数据集 [CASMI NP Frozen FPNet View Cache 20261005](https://www.kaggle.com/datasets/xiaoyuzhoux120/casmi-np-frozen-fpnet-view-cache-20261005) Version 1 已就绪，Data Explorer 显示 5.27 GB；自动同步关闭，来源保留原 Notebook。未经过本地下载/上传，不修改浏览器权限。下一版换用该数据集，仍在执行时逐文件核验缓存；页面可用不等于二进制核验已完成。

因 Version 1 精确 worker 时间不可得，后续预算保守预留 600 秒给该次挂载失败，明确这不是实际测量值。新的 14400 秒预算不再从零启动；旧 10528.1 秒另行保留。科学参数及模型源不变，重打包只更新恢复声明和预算来源。

## 第二次启动：环境不兼容，实际 16.8 秒

缓存重训 Notebook Version 2（355532778）已成功挂载私有数据集并校验 7 个化学输入文件，但新 Notebook 的初始容器为 Python 3.13.15。它无法安装冻结的 CPython 3.12 RDKit 2026.03.3 wheel，故在训练前停止。页面实际 worker 耗时 16.8 秒，零训练 seed、无 GPU 预检、无验收或比赛提交。此前页面的约十分钟 Running 包含等待启动，不能当作模型训练耗时。

原 FPNet Notebook 编辑器明确保留 `Pin to original environment (2026-06-30)`。恢复将在该 Notebook 保存新版本，复用原 Python 3.12 环境和私有缓存，保留原失败 Version 2 与独立缓存 Notebook 两次失败。模型与科学参数保持一致；实验预算扣除保守挂载预留 600 秒与实际 ABI 启动失败 16.8 秒，共 616.8 秒。

## 第三次启动：原固定环境，真实 GPU 预检通过

使用原 FPNet Notebook 的 Version 3（355536163），冻结源提交 `2e16df4`，2026-06-30 固定环境、Python 3.12、Torch 2.10.0+cu128、两张 Tesla T4、Internet OFF。两个代码单元均通过页面全选复制逐字核对。

- 私有缓存真实核验：训练 119,977 身份、开发 1,930 身份，74.912 秒；不重建缓存，不复用失败权重。
- 真实架构 GPU 预检：64 样本、两卡、强制溢出跳步保护，FP32 恢复与直接更新权重差 0，恢复实际更新 1 次，10.533 秒；预检权重丢弃。
- 已见种子 20261004、第 1 epoch、1,100 次真实优化器更新。此条只证明启动与训练进展，不代表三 seed 完成、M2 门槛通过或比赛提交。

正式日志：https://www.kaggle.com/code/xiaoyuzhoux120/casmi-np-pairtail-fpnet-20261004?scriptVersionId=355536163

## 第三次运行：预算退出缺陷，19 个完整轮次保留

页面实际 worker 用时 **2840.0 秒，failure**。20261004 种子完整完成 epoch 1–19；epoch 19 的累计真实更新 35,625，训练 BCE 0.28182085、开发 BCE 0.28790706。8 次 AMP 梯度溢出通过 FP32 全有效批次重放恢复成功。epoch 20 途中触发冻结的单种子 2700 秒硬上限，代码抛异常而未结束该种子并继续其他种子；并非非有限权重、数据泄漏或 Kaggle 公榜出分。

预算退出应在至少完成六轮后，按已观测近三轮最大耗时 +10 秒预留，在完整 epoch 边界结束；不能延长 45 分钟上限、不能修改学习率/最大轮次，也不能把未完整 epoch 20 当完成。新修复 17 项相关测试通过。新实验需承接已耗 worker 2840.0 秒与前两次启动预算扣除 616.8 秒，累计 3456.8 秒；更早实验 10528.1 秒继续另列。

正在下载真实模型输出并核验可恢复的完整 seed。页面日志不等于模型二进制已核验；尚未完成三种子训练、M2 排序验收、独立验收或本轮比赛提交。

## 第四次运行：二进制 GPU 重放通过，继续剩余种子

正式 Version 4（355568739），冻结源 `05f2095`，固定 June 30 环境、T4 x2、Internet OFF。原 V3 模型输出直接作为不可变 Notebook 输入挂载成功；未再次使用本地上传。核验声明列出的 7 个文件 SHA、原源码、基础拆分及选中 epoch 后，实际 GPU 重新预测：单谱与融合 logits 对存档的最大差均 **0.0**；重新计算 BCE 0.28790709376335144，对原 0.28790706396102905 的浮点差约 3e-8。保留完整 epoch19 权重与真实更新 35,625；不恢复优化器、不使用半轮 epoch20。剩余种子 20261005、20261006 从头训练。

已完成 30 项相关工程测试，包括 37/400/800 查询动态数量、ID 重映射输入不变、目标列不进入运行特征、C0真实Top1保护、溢出批次恢复和四种子排序器独立进程。测试通过不等于真实排序收益或独立验收通过。

## 后续开发排序阶段准备（尚未执行）

M3 独立封装完成：冻结训练身份中按固定散列顺序取前 4,000 个查询，标签盲的最多 256 个负候选，4 个随机种子的 LambdaRank 和预先指定的深度/叶节点网格；开发集验证 C1 指标必须先逐项复现，再选择 C3。C4 仅增加 LOTUS 候选和来源标记，继承 C3 的真实权重；C6 对真正的 C0 原始排名应用 PairTail，不能误将 C1 第一名当成受保护的 C0 第一名。外部 forward 权重尚不满足域、许可及身份核验，保持关闭。

实际 LOTUS 全源处理得到 149,436 个默认互变异构归一后的中性连通身份，未使用任何查询窗口或标签；CC BY 4.0 署名随资产保存。官方 LightGBM 4.7.0 MIT wheel 带完整哈希。私有 Kaggle 数据集 **CASMI NP LOTUS Ranker Assets 20261005 / Version 1** 创建成功、Data Explorer 显示 17.78 MB 解包输入。第一次使用 Release CDN 的创建内部错误已保留；改为固定 Git 提交的稳定 raw 文件地址成功。它不包含答案表、查询 ID 或模型权重。

31 项本地检查通过，包括标签污染不会改变特征、37/400/800 个运行时 ID 的改名不改变谱图输入、真实四模型 CPU 排序管道、C0 第一名保护及预算边界恢复。检查通过仅代表工程管道验证；M3 尚未获得真实开发指标、验收和比赛分数。

### 发布链路已准备，尚未执行验收或提交

开发、一次封存验收、纯推理发布分别封装为独立 Notebook。开发成功的输出保留三个不可变的选入 FPNet 模型，下一版本只携带冻结排序模型、候选来源及规范身份缓存，避免在同一 Notebook 中挂载多个历史版本。M5 在读取验收文件前必须通过数据、源代码、排序模型和 FPNet 选择清单的 SHA 核验，并建立一次性打开记录；M6 必须持有同一候选冻结哈希的实际通过凭证。比赛重跑入口执行谱图到指纹和候选排序，禁止重新拟合或计算封存标签成绩。

为保证 C0/C1 复现，参考谱图仍只采用原 foundation 规范身份缓存中的身份资格；额外候选的规范化缓存独立保存。查询只读谱图和测量元数据列，主动丢弃可见文件可能包含的结构、分子式和 InChIKey 标签。Library 指纹缓存按完整 SMILES 别名和规范身份作键，相同输入的特征逐位不变；不会只按可能忽略同位素差异的短身份合并特征。

**55 项相关检查通过**。包括封存比较门槛、科学协议与打包、输入划分、GPU 工程修复、运行时标签列隔离、动态 ID、输出规范身份去重及真实 CPU 四模型排序管道。三种后续封装目前均为 prepared-not-executed，不能将测试结果当作实测开发、验收或公榜收益。当前 Version 4 第二种子实际处于第 11 轮、20,350 次权重更新；第三种子尚待启动。

### PRD 模块回退规则在开发执行前冻结

PRD 第 10 节要求某模块未达局部门槛时，回退前一通过模块继续后续组合。因此在未取得任何 M3 开发结果之前补全以下流程：若 4 个排序配置均未通过 C3，则保留 C1 的原排序、禁用依赖合格学习排序模型的 LOTUS C4，继续同一个预设 C6 的 FPNet / RRF / PairTail 网格。不会使用被开发保护规则拒绝的排序权重。封存与发布代码显式区分 ranker-enabled 和 C1 fallback；无论哪条路径，最终 C6 开发、独立验收及预算门槛均不放宽。

### 最新实际训练与草稿启动成本

Version 4 第二种子 20261005 已完整结束 20 轮、37,500 次更新，最后开发 BCE 0.2886069715023041；第三种子 20261006 在本次日志快照为第18轮、32,875次更新。尚无完整三种子门槛结果，不能发布或提交。

编辑下一阶段说明时意外启动了草稿会话，未执行任何代码，已通过页面 Stop session 关闭。实际 worker 时间无法确认；为避免漏记成本，在 M3 准备时间及总预算中保守计入 300 秒。该扣除不延长模块或总预算；正式 Version 4 作业未受影响。

### 三种子训练已结束，开发候选门槛正在计算

实际日志显示第三种子 20261006 完成 20 轮、37,500 次更新，最后开发 BCE 0.2864888608455658；第19轮 BCE 0.28647443652153015 更低，实际最终选择仍以下载后的选择清单和模型二进制为准。两种子 AMP/FP32 恢复累计分别9次与10次，日志均确认更新成功。作业进入同一质量窗口的 FPNet 与随机排名对照；首个已见进度为25/1930查询、7444缓存指纹。完整三种子输出和局部门槛尚待完成并下载核验，独立验收未打开、未提交新比赛版本。

### 开发执行前的训练特征计算优化

M3 实测结果尚未产生。在代码审计中发现原实现先计算全部训练候选的碎片特征，再依据仅使用 library_score / fp_dot 的固定标签盲规则取最多256行用于拟合。改为先按完全相同的规则确定这些行，只为会进入训练的行计算昂贵碎片解释；保留全候选池的 FP 标准化、排名、analog 得分和训练 margin。未选中行的碎片占位不进入拟合，选中行不变；开发、验收和比赛推理仍计算全部候选。查询固定的仪器、极性、碰撞能量统计及清洁谱峰移到候选循环外，避免重复计算。

新增600行测试确认两种路径都选中完全相同的256行、选中训练特征矩阵逐元素一致，省去344行碎片解释，被遗漏真值不会插入。独立加载原提交代码，对混合加合物/极性、缺失仪器和能量等输入验证完整评估43列特征逐元素相同，记录见 feature_runtime_equivalence.json。56项相关检查通过。没有改变候选池、科学参数、模型、选择网格或验收门槛；这仍是尚未实际执行的 M3 封装，不宣称真实加速率或提分。

### Version 4 实际成功及独立复核

页面确认 Successfully ran in 8603.7s。实际输出ZIP 900,409,985字节、49文件，SHA256 b8e97ca27c442237dc12e778671c76d38ebf933859248d1e311f7568bcd774db。三个选入模型轮次为19/20/19，选入权重真实更新35,625/37,500/35,625；完整训练轮次19/20/20。三模型权重和logits均有限，集成logits逐元素等于三种子的平均，归档10份源代码哈希均匹配冻结源。

集成开发BCE 0.28146469593048096，训练位频率常数基线0.5708246827125549，通过10%门槛。同窗候选真值存在子集1912/1930，融合视图相对随机排序MRR差0.4992470854917024，10,000次配对bootstrap 95%CI [0.4801517145717654,0.517342501649246]，通过门槛并选择fused。独立复核所有案例排名、随机置换、RR和两组区间一致。它不代表Kaggle分数，也不能证明隐藏测试候选真值覆盖。

累计研究成本12030.149826481998秒，原4小时预算余2369.85秒；下一阶段仍计入300秒草稿启动保守扣除及真实准备成本。当前只有模型与开发报告，没有submission.csv，独立验收仍封存。完整输入训练行与候选池未下载重建，仅由实际资产哈希绑定；独立复核限于保留的二进制、logits、排名案例和源代码。

### M3 正式启动

已实际更新自身模型输入Version3到成功Version4，两个代码单元与冻结封装逐字一致。实际Saved & Run All Version5 / 355588043，名称 Frozen M3 PairTail dev a8d6c38，T4x2、原June30环境、Internet OFF。该阶段复用三个训练权重，不重复FPNet训练；仍需开发候选冻结、一次独立验收与纯推理输出才能提交比赛。目前不是比赛提交，没有新公榜分数。

### M3 Version5 工程失败及修复

正式日志151.5秒失败：foundation训练集保留了未知/多电荷加合物，而开发集已限定支持的单电荷质量域；新排序器直接把训练谱图送入严格查询质量函数。未读取验收，未拟合排序模型。修复只让原先固定4000身份中的支持单电荷、有限正中性质量测量进入检索拟合；不使用分子式/结构标签反推查询质量，不替换或补齐失去全部合规测量的身份，保留排除行/加合物/身份审计。FPNet仍复用原模型和预测视图；开发、验收及推理域保持原规则。26项检查通过。失败151.5秒追加计入原研究预算，300秒草稿扣除也保留。当前没有新submission.csv或比赛分数。

修复正式保存并运行Version6 / 355588750，名称M3 charge1 repair 38a64ce，复用自身Version4模型输入。两个编辑代码单元与本地冻结封装逐字一致。嵌入源码SHA256 45de8e5dcfba736b9617a156b1907a77082bf4218ff31c7272cefae37a926c34；扣除草稿300秒及V5失败151.5秒后、当前准备之前余1918.35秒。不重复训练FPNet，仍不是比赛提交。

Version6实际运行约1932.3秒后按冻结研究预算停止，日志最后显示结构池身份审计48000/134019。没有创建candidate_freeze，没有打开一次性验收，没有生成submission.csv，也没有提交比赛。该版本的失败原因是CPU结构身份审计耗尽剩余预算；版本5的加合物错误和151.5秒成本均保留。可提交的既有结果仍为V1回退0.176。

用户要求重新开始。为解决Version6的主要瓶颈，只改进结构身份审计的执行方式：使用确定性有序的线程池调用RDKit canonicalization，避免每个短生命周期进程重复导入RDKit；训练/验收数据、候选池、模型、排序参数和门槛均不变。29项相关测试通过，当前已重新封装Notebook，尚未保存或运行新的Kaggle版本。

已实际保存并启动Version7（脚本355967631，名称M3 threaded restart 38fefb0），GPU T4x2、Internet OFF、仍复用Version4三模型。当前正在运行，尚无新输出或比赛分数。

Version7实际于1930.8秒失败。线程化身份审计有效：134019/134019全部完成，用时约1672.8秒；随后参考谱扫描达到819200行、保留754579条，在错误沿用的旧实验剩余额度处触发`Reference construction research/module budget exhausted`。没有拟合排序器、没有打开验收、没有submission.csv。失败输出保留完整24.86 MB `base_structure_pool.parquet`及其哈希清单、身份映射和三模型logits。

用户随后明确要求继续。新封装只复用Version7中哈希有效、范围标记为训练/开发质量窗并且未接触验收/测试的结构池；其余失败运行的模型或排名输出均不复用。旧M3失败成本4314.6秒继续单独记录，新M3获得明确的7200秒独立额度。候选规则、模型、排序参数和验收门槛不变；25项相关测试通过。尚未启动新版本。

### Version 8: cached fresh-budget restart

- Started 2026-10-07 14:38:49 America/Los_Angeles as `M3 cached fresh budget ab47cc6`.
- Kaggle scriptVersionId: `356186436`; T4 x2; Internet OFF; pinned 2026-06-30 environment.
- Both editor code cells were copied back from the live editor and matched the generated notebook byte-for-byte before launch.
- Self-notebook input reported no pending update and is expected to expose Version 7 outputs. The runner accepts only the complete ranker pool whose manifest scope and SHA-256 both match; it reuses no model, ranking, acceptance, or test output.
- This run has a fresh independent 7200-second M3 allowance. The 4314.6 seconds spent in prior failed attempts remain in the cumulative audit only.
- Status at launch: running; no acceptance or submission has been opened.

### M3 Version 8 预算失败与 Version 9 工程修复

Version 8 / `356186436` 已在独立 7,200 秒 M3 额度边界内失败，异常为 `TimeoutError: M3 candidate feature budget exhausted`；未打开验收集、未生成 `candidate_freeze.json`、未生成比赛提交。它没有命中预期的 Version 7 结构池挂载，因而重新完成了 558,889 行结构池及完整参考谱扫描；参考扫描后训练 unknown 特征仅完成 400/3,921 个身份、135,005 个候选。日志证明瓶颈是 CPU/RDKit 的候选标准化、指纹和化学特征，不是 T4 算力不足。旧页面显示的一万多秒是累计审计时间；实际新 worker 在 7,220 秒处按冻结预算退出。

修复只改变确定性工程执行：从 Version 8 严格绑定的失败输出复用完整、标签盲的 `reference_rows.parquet`、`reference_metadata.parquet`、`reference_spectra.npz`；不复用任何部分特征、排序器或排名。候选池仍按哈希和 scope 检查。RDKit tautomer enumerator 在同一进程内复用，保持 `canonical_target` 结果逐位一致；训练候选的诊断离子/中性丢失 SMARTS 只为原本会进入最多 256 行标签盲训练提案的候选计算，并缓存 `(identity, rule_id)` 匹配，开发/验收/推理仍对全部候选计算。候选池、训练身份、标签盲选择、43 项特征、四种子 LambdaRank 网格、C0/C1/C3/C4/C6 规则和验收门槛均未改变。

67 项相关测试通过；500 个真实池结构的共享 enumerator 基准保持与原 `canonical_target` 完全相同的指纹。新封装 SHA256 为 `ae00e0a8601aa9d625093f0e79323040d8f6dc6b40b7ee6d64d823b8969a8a1a`。Version 9 尚未启动，仍须先把自身 Notebook 输入固定到已完成的 Version 8 输出并逐字核对两个代码单元。

### M3 Version 9 reference-cache repair started (2026-10-08)

- Version 8 ended at 7,220 worker seconds with the sealed acceptance still unopened after 400/3,921 unknown-training identities. Its complete reference scan was retained; no partial candidate features, models, rankings, or acceptance artifacts are eligible for reuse.
- Engineering repair commit `a29408b` loads only the exact Version 8 reference cache, reuses one exact RDKit tautomer enumerator, caches repeated diagnostic-rule matches, and delays training-only chemical evidence until after the frozen label-blind negative selection. Scientific parameters and all evaluation/inference features remain unchanged.
- The relevant suite passed (`67 passed`), the notebook was regenerated with embedded source SHA-256 `ae00e0a8601aa9d625093f0e79323040d8f6dc6b40b7ee6d64d823b8969a8a1a`, and GitHub was updated before execution.
- Formal Version 9 is running as `M3 cached reference repair a29408b`, scriptVersionId `356616700`, with GPU T4 x2, Internet OFF, pinned 2026-06-30 environment, and the latest self-input (Version 8). No acceptance or competition submission has been opened.

### M3 Version 9 failed: failed-run cache was not mounted (2026-10-09)

Version 9 / `356616700` failed after about two hours with the same frozen `TimeoutError: M3 candidate feature budget exhausted`. The expected `M3_reference_cache_loaded` event is absent. Instead, the visible log shows a fresh `M3_reference_scan` followed by only 425/3,921 unknown-training identities and 147,276 candidate rows before the deadline. The acceptance set remained sealed; no candidate freeze or submission was created.

The engineering diagnosis is now concrete: adding this notebook as its own input exposed the last successful output, not the selected failed Version 8 output containing the completed reference cache. Therefore the Version 9 loader correctly found no eligible `reference_rows.parquet` bundle and repeated the expensive scan. This is an input-artifact delivery failure, not a GPU-training failure and not evidence about the frozen scientific parameters. A further run must first publish the failed-run reference bundle as an independently mounted, hash-bound private dataset (or obtain an equivalent supported Kaggle artifact mount); it must not start another M3 run against the current self-input.
