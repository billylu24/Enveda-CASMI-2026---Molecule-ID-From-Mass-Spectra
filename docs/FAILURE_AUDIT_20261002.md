# 失败机制与化学规则覆盖审计（2026-10-02）

## 从实际输出能确定什么

本轮分析读取正式Kaggle输出的32分子`behavior_check.json`及当前ChEBI/LMSD联合目录。
行为文件SHA256为`60889d3c1071d7e4f97ef7d17da8f0c69634a6f399986c852866a533e6d31f6d`。
这批分子此前已经观察过，按原始结构键字母序选择且只取第一张源谱；分析不构成独立验证。
当前本地没有训练全集、这些查询的原始谱图/加合物/前体元数据或完整COCONUT目录。

| 可核实项目 | 实际结果 |
|---|---:|
| 行为查询 | 32 |
| 被0.5置信度guard保护 | 23 |
| protected查询的Top1错误 | 23/23 |
| protected查询的Top25缺失 | 8/23 |
| 真正执行过化学证据抽取的low-confidence查询 | 9 |
| 新公共目录中的真值原始键覆盖 | 32/32，均为中性代表 |
| 对应公共代表满足五条规则之一的SMARTS | 2/32 |
| 扩库后倒数排名发生变化 | 2个，均退步 |

23个protected查询的MRR@25为0.110573；9个low-confidence查询的原MRR为0.211111，
扩库后为0.201852。这里的“guard错误”表示它保护了错误首位或缺失真值的排名，
不表示关闭guard就必然能纠正，也不能据此在这32个已观察样本上选择新阈值。

32个真值原始键都在新公共目录，可以排除这批查询的“新源库完全没有原始真值键”。
**不能**由此推断真值通过查询质量窗、进入实际有效候选池或被COCONUT历史分支使用。
当前日志没有完整池、质量中心、加合物、谱图或首二名相似度，Top25缺失的进一步原因无法辨认。

## 具体已知真值案例

名字来自公共目录的ChEBI注释；原始查询图未保存，公共同键代表不能冒充原始查询SMILES。
排名来自行为输出对候选做互变异构体规范化后的实际评分。

| 原始查询键 | 公共注释 | 置信度 | Guard | A→B真值排名 | 已确认的问题 |
|---|---|---:|---|---|---|
| ACNHBCIZLNNLRS | paxilline | 0.603113 | 是 | 缺失→缺失 | 新库有真值，但guard阻止扩展；质量资格/截断原因未知 |
| ADEBPBSSDYVVLD | donepezil | 0.888466 | 是 | 3→3 | 真值已经输出，但错误首位被保护 |
| AIONOLUJZLIMTK | hesperetin | 0.930373 | 是 | 4→4 | 很高的谱图相似度也不等于正确结构首位 |
| BBNQQADTFFCFGB | purpurin | 0.215221 | 否 | 4→5 | 扩库竞争挤压已有正确候选；没有新增召回 |
| CKLJMWTZIZZHCS | L-aspartic acid | 0.017486 | 否 | 5→6 | 扩库竞争降低倒数排名 |

purpurin的评分规范化键为`CHERCCPWQNZOOH`，不同于其原始键`BBNQQADTFFCFGB`。
因此候选覆盖、参考排除和评分需要同时保留raw key与规范化身份，不能只比raw key。
来源名`enveda-np-examples`也不能代替实际化学结构检查，例如其中有donepezil这一公共注释。

## “规则0命中”不能直接归因于化学知识无效

`HybridChemistry.variants()`仅在`not protected`时调用`extract_evidence()`。
23个protected查询的空证据是**没有执行抽取**，不是检查后证明谱中没有诊断信号。
真正接受抽取的是9张low-confidence查询谱。

五条规则的结构范围是磷酸胆碱、磷酸乙醇胺、O-己糖、O-葡糖醛酸和硫酸酯。
当前32个公共同键代表只有以下两个满足其中的SMARTS，均为O-己糖规则：

- `BJRNKVDFDLYUGJ`：hydroquinone O-beta-D-glucopyranoside；置信0.413202，未保护，真值rank5。
- `CJHYKSSBQRABTM`：ChEBI注释为含葡糖基的三羟基齐墩果烯酸；置信0.772723，被保护，rank12。

其余30个代表不在目前规则的结构范围内。没有匹配不证明这些分子的谱没有其他可解释碎片。
第一张源谱也不代表该分子所有碰撞能量/模式下的碎裂；缺失信号不能用于剔除候选。

既有训练预处理的离子模式类别是标准`positive`/`negative`，没有发现全局编码错误证据。
但32个查询具体的mode/adduct未保存，不能证明它们各自满足规则适用条件。
生产匹配要求mode和adduct字符串精确相等：`Positive`、`pos`或前后空格会阻断匹配；
铵、钾、脱水和多电荷加合物也可能不属于当前规则支持范围。
新审计会报告原始编码、适用计数、峰质量匹配、弱峰门控，并独立计算别名/空格修复后的
诊断匹配数。这个反事实只用于定位编码问题，不会改生产输入或预测排名。

另一个静态风险是`baseline.load_candidates()`预筛只用35ppm，随后历史检索用
`max(35ppm, 0.006Da)`。低质量区域可能提前丢掉后续质量窗本应允许的参考谱。
这是代码路径的不一致，尚未量化为以上案例的实际错误来源。新审计采用统一35ppm/0.006Da。

## 可复用接口与真实holdout接入

`casmi_ml.failure_audit`不创建新split、不追加真值候选、不训练模型、不再次扫描reference。
复用GAN实验已冻结的官方规范化身份/骨架划分：train60K、dev1000、accept1000，
以及该实验对NP来源/可见400分子的预定排除。来源隔离NP行为诊断需使用独立命名空间，
保持“此前观察过、非独立、不用于调参”的标记。

调用方负责unknown-spectrum场景排除全部查询raw aliases及规范化同身份reference，
并从library候选移除heldout身份；已有public结构可以保留。库池由library、COCONUT、
ChEBI/LMSD组成，冻结质量、表示和电荷政策；不能通过heldout答案补候选。

```python
from casmi_ml.failure_audit import prepare_catalog, audit_query, save_audit

# full_pool是完整、已执行上述排除和电荷政策的统一候选池。
# 若已有identity/canonical_identity列会复用，避免每查询重复计算。
full_pool = prepare_catalog(full_pool)
cases = []
for molecule_id, group in holdout.groupby("molecule_id", sort=True):
    case = audit_query(
        group, full_pool, baseline_pairs[molecule_id], rules,
        confidence=confidence[molecule_id],
        query_truth_smiles=group.iloc[0].normalized_smiles,
    )
    cases.append(case)
report = save_audit("failure_audit.json", cases, metadata=frozen_protocol)
```

`baseline_pairs`可为`(key, SMILES)`或`(key, SMILES, score)`。不要只传质量过滤后的小池，
否则无法把源池缺失与质量过滤失败分开。已有规范化identity列必须使用同一RDKit评分政策。

每个case输出：完整池oracle覆盖、library/public来源覆盖、质量资格、Top25真实排名、
保护错误及其具体范围、全部查询独立化学抽取、mode/adduct门控和truth motif覆盖。
机制分为源池缺失、不支持/错误加合物、质量过滤错失、排序/Top25截断、真值非首位及正确首位。
按已存在的`split`字段聚合指标，可并排查看train/dev/accept；训练验证差距必须来自相同
评分/候选/查询政策的实际模型结果，当前32例不能替代该比较。

只处理现有导出结果：

```sh
.venv/bin/python -m casmi_ml.failure_audit outputs \
  --behavior kaggle_release_chemistry/run_output/behavior_check.json \
  --catalog external/chemical_catalogs/structures.parquet \
  --dictionary configs/chemical_priors.json \
  --output artifacts/failure_audit/observed_behavior.json
```

在Kaggle或已恢复的数据环境审计既定holdout及其模型输出：

```sh
python -m casmi_ml.failure_audit queries \
  --queries prepared_accept.parquet --pool frozen_full_pool.parquet \
  --rankings baseline_rankings.json --dictionary chemical_priors.json \
  --output accept_failure_audit.json
```

rankings JSON是列表，每条含`molecule_id`、`pairs`、`confidence`。对baseline、common、
control、GAN分别调用同一审计/评分，保留相同候选池和split；不要从accept案例调整规则或权重。
当前本地8项相关测试通过，覆盖库/质量/排序分解、低质量绝对窗、规范化别名、wrong guard、
protected查询独立规则抽取、编码/弱峰/加合物门控，以及split聚合与输出边界。
