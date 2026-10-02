# CASMI26 GAN 正式运行耗时审计（2026-10-02）

本报告只审计运行工程问题，不调整本轮科学参数。对象是正式 notebook Version 2，冻结源提交 `a45c65e9f16501c554dfd4a12679003d630cb0ff`，嵌入源码 SHA256 为 `ee7ccca6b373156e81522dea9598ccf4aeae1fe4eae1c4f827bc4383b4a2f93d`。

审计时 V2 仍健康运行，日志已完成 991 个 development 分子的排序，unknown acceptance 推进到 200/994；这只是当时的进度快照。没有已完成的 GAN 独立验收成绩，也没有新 GAN 比赛分数。两臂模型训练合计约 180 秒，当前主要等待的是后续结构处理、图编辑和候选评估。以下建议供后续工程版本使用，不据此中断或修改正在运行的冻结 V2。

## 当前顺序和剩余工作

`casmi_ml/adversarial_release.py:416` 准备数据，`:417` 训练两臂，`:418` 重新加载权重。接着依次执行：

| 阶段 | 已核查源码 | 说明 |
| --- | --- | --- |
| Development | `casmi_ml/adversarial_release.py:438` | 独立查询逐分子排序，当前日志已完成 991 个 |
| Unknown acceptance | `casmi_ml/adversarial_release.py:439` | 独立未知参考条件验收，当前队列 994 个 |
| 写出前两组汇总 | `casmi_ml/adversarial_release.py:440` | 在 unknown 验收返回后才写 development 汇总；尚无输出不能推断 development 尚未完成 |
| Known acceptance | `casmi_ml/adversarial_release.py:448`、`:456` | 从同一验收查询中选择仍有真实同身份参考的子集；先剔除所有查询及其分箱向量副本，不人工补入真值候选 |
| 测试推理 | `casmi_ml/adversarial_release.py:475`、`:478`、`:485` | 再扫描训练参考、建立索引，然后按实际 test 分子分组推理；本次可见集为 400 个分子 |

known 子集大小须由实际输出核实，不能默认等于全部 994 个。不同阶段的置信度和图编辑触发比例可能不同，不能把当前 unknown 的速度直接当作所有剩余阶段的固定耗时。

## 为什么更强 GPU 不能直接解决本次等待

训练器在 `casmi_ml/adversarial.py:582` 选择 CUDA，但 `load_generator` 的默认设备是 CPU（`casmi_ml/adversarial.py:549`）；正式 release 在 `casmi_ml/adversarial_release.py:418` 未覆盖该默认值。因此重新加载后的指纹预测也走 CPU，后面的 RDKit 处理和 NumPy/稀疏检索更不会因选择两张 GPU 自动并行。

两臂候选计算已经共享：`casmi_ml/adversarial_release.py:353` 先预测两组指纹，`:360` 对同一查询只调用一次 `engine.rank`，`:239` 至 `:274` 建出的图编辑池供两臂共同使用。不能把“合并两臂候选计算”当作尚未实现的两倍提速方案。`torch.set_num_threads(4)` 也不等于把逐分子 RDKit 循环分发到四个工作进程。

## 可减少的重复工作

### 统一规范结构缓存

准备阶段的 `canonical_structure` 已返回官方式身份、规范 SMILES、质量、分子式和电荷（`casmi_ml/adversarial.py:49`、`:56`、`:61`），并在 `:173` 至 `:181` 为目录保存这些字段。release 的 `identity` 却在另一套独立缓存中再次 `Canonicalize`（`casmi_ml/adversarial_release.py:33`、`:38`），`canonical_fingerprint` 又单独规范化同一结构（`:41`、`:46`）。

后续可复用不可变结构记录，并在记录中增加规范指纹。持久化缓存键至少包含输入源 SHA256、RDKit 版本、互变异构体规范化配置、指纹配置和原始结构字符串，避免环境变化后误用旧值。当前官方身份以规范图的 InChIKey14 定义，不能用原始 raw key 替代。

必须保留两种指纹语义：神经目标及神经候选使用规范图指纹；历史 `hybrid.py` 的 analog 指纹使用其原有图表示。统一缓存接口不代表可以把历史指纹换成规范指纹，否则会改变历史控制臂和路由结果。

### 避免生成图的重复规范化

`casmi_ml/graph_candidates.py:76` 开始每锚最多 256 次尝试，`:110` 对合法图求官方规范身份；返回结果已经包含规范 SMILES 和身份。release 又在 `casmi_ml/adversarial_release.py:253`、`:254` 重算身份和规范指纹。

可复用已验证的生成图身份及规范 SMILES，再计算相同定义的指纹。进一步可缓存确定完全相同的图，或按原随机序列惰性遍历，直到取得原协议的 32 个合格候选或耗尽全部尝试。不能只计算前 32 个原始编辑：其中可能含非法图、重复图和目录已有身份，原实现还会继续寻找后面的合格候选。

优化必须保留原随机序列、锚顺序、排除集合、输出顺序，以及连通性、原子度数、电荷、分子式、质量和官方身份去重检查。图编辑缓存的键含查询内容派生的 seed（`casmi_ml/graph_candidates.py:58`、`:165`；release 的 seed 来源在 `casmi_ml/adversarial_release.py:82`），不同查询共享同一锚也未必命中。删除查询 seed 会改变科学协议，不能作为缓存优化。

### 共享纯结构特征，保持查询条件隔离

每个新 `Ranker` 在 `casmi_ml/adversarial_release.py:168` 至 `:170` 重建三个指纹缓存；unknown、known 和 test 阶段可共享不依赖查询真值的结构特征。独立 public 池也可预处理一次，当前 `build_pool` 在 `:434`、`:453`、`:478` 重复调用。

这些阶段的参考排除规则不同。可共享纯结构信息，但不能复用已经按某个阶段排除过的参考集合、最终排序或最终候选表。known 条件必须继续删除验收查询的所有分箱向量副本（`:448`），unknown 条件必须继续屏蔽官方身份及 raw aliases（`:425`、`:433`）。

其他较小的重复点包括：

- `hybrid.py:32` 每查询重建 `masses[order]`，可在索引建立时保存排序质量数组。
- `casmi_ml/adversarial_release.py:178` 每查询重算至多 25 个参考结构指纹，可按其原指纹定义缓存。
- `casmi_ml/chemical_priors.py:141` 至 `:155` 重新解析候选并检查 SMARTS。可按结构与规则 SHA 缓存布尔 motif 向量，仍按当前查询的证据强度和权重计分。
- `casmi_ml/adversarial_release.py:232` 和 `:273` 在加入图候选前后重算原有候选的化学分数，可只扩展新增图的结构支持结果。
- `casmi_ml/adversarial_release.py:162` 为各阶段重建参考稀疏矩阵，可考虑一份 union CSR 加阶段专属、已审核的有效行索引。

## CPU 并行的正确性边界

准备结构循环（`casmi_ml/adversarial.py:173`）和查询评估循环（`casmi_ml/adversarial_release.py:357`）目前都是串行。先并行小批量纯结构特征预计算更容易控制内存：每个工作进程持有独立 RDKit enumerator，结果按输入顺序归并。

逐查询并行需要共享只读参考索引，或采用有明确内存预算的分块任务；不能让每个工作进程复制约百万行的谱图字典和稀疏矩阵。必须保持查询内容派生 seed、稳定 tie-breaking、身份过滤和 Top25 次序一致，并记录实际耗时和峰值内存。多线程是否对具体 RDKit 操作有效应先测量，不假定线程数增加就有线性提速。

## 研究 notebook 和比赛推理应在后续版本拆分

当前 `kaggle_release_gan/build_notebook.py:48` 直接调用完整 `run`。`casmi_ml/adversarial_release.py:416` 至 `:456` 无条件执行准备、双臂训练、development、unknown 和 known 验收，之后才产生 test 提交。这适合一次研究实验，但若比赛重新执行同一 notebook，会重复整套研究工作。

`seconds_per_arm=1800`（`casmi_ml/adversarial_release.py:417`；检查位于 `casmi_ml/adversarial.py:611`）只限制每臂训练。源码没有覆盖准备、结构编辑和验收的总运行时上限，也没有这些阶段的续跑机制。本报告未核实官方比赛执行时限，因此只指出重复计算风险，不断言隐藏重跑必然超时。

后续 inference-only 发布版本应：

1. 从已完成、可核验的冻结实验加载 `model.pt`、训练拟合的预处理、目录和必要索引；记录权重自身 SHA256、训练源提交、嵌入源码 SHA256、训练 manifest、预处理 SHA256 以及所有输入文件 SHA256。不得用验收结果再选科学参数。
2. 保持本轮架构、指纹定义、训练所得预处理、路由阈值、融合权重、化学规则、质量窗口和图编辑协议。先将工程拆分与科学改动分开验收。
3. 针对同一冻结 checkpoint，在原推理路径和新入口之间逐查询核对参考有效行、候选身份及顺序、生成图、神经预测、各臂排序和最终提交，解释浮点容差；候选和 Top25 的身份/次序必须一致。
4. 每次从实际挂载的 test 重新读取谱图、分组、检索和生成提交。隐藏 test 可改变分子数量、ID、谱图、质量、仪器或模式，不能固定 400 行，不能按可见 test 的 molecule_id 查表，也不能复用可见 `submission.csv` 作为隐藏预测。

还需明确版本等价性的范围：现完整流程把当前 `test.parquet` 传入准备函数（`casmi_ml/adversarial_release.py:416`）。准备阶段读取可见结构排除信息和谱图签名（`casmi_ml/adversarial.py:168`、`:169`），并在 `:219` 至 `:258` 将匹配的官方身份排除出神经训练分组。隐藏 test 若变化，完整重训时的排除集合也可能变化。

因此固定现有权重的 inference-only 应作为明确冻结训练状态的新发布版本。其逐查询等价验证对象是“相同 checkpoint 的原 Ranker 推理”，不能声称它在所有变化后的隐藏 test 上都严格等价于“重新准备并重新训练”的原 V2。冻结训练 manifest、可见排除来源和模型来源有助于解释这一边界。

当前 V2 应继续获取真实验收输出和比赛提交结果。完成后再按本报告实施与验证工程优化，不改变本轮结果的归因。
