# 化学证据与公共结构库接入（2026-10-01）

## 已核实的现状

历史最佳公开成绩仍为0.176。60K残差模型已经使用了 `precursor_mz-fragment_mz`
的数值中性丢失特征；此前缺少的是把诊断离子/丢失与候选化学结构联系起来。
COCONUT与PubChemLite已有接入。本轮新增ChEBI、LIPID MAPS结构导入器，以及
PubChem离线子集的通用接口。未下载PubChem全库，也未运行新的比赛提交。

官方CASMI Data页只列train.parquet、test.parquet、sample_submission.csv，
目前未核实主办方发布了专门的诊断词典：
https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data

## 两层资源不能混为一谈

1. MS-FINDER公开TSV：2,995条产品离子、1,643条中性丢失。适配器验证公式质量，
   显式处理产品离子的电子质量修正、正负模式与丢失差值误差传播，输出证据注释。
   上游表没有直接的SMARTS、加合物或碰撞能量标签，所以不自动推断官能团。
   文件专属许可未确认，资源只保存在被Git忽略的external/msfinder，不进入发布包。
2. `configs/chemical_priors.json`：项目手工编写的五条文献支持软规则。
   磷酸胆碱184.0733、磷酸乙醇胺丢失141.0191、O-己糖残基丢失162.0528、
   O-葡糖醛酸残基丢失176.0321、硫酸酯丢失79.9568。每条包含精确质量、
   模式、加合物范围、SMARTS、来源与局限。它们不是主办方官方词典。

H2O、NH3、CO2丢失不是专一官能团诊断。本轮不使用它们做硬筛选。
特别是铵加合物失NH3、甲酸加合物失HCOOH不能直接解释为分子官能团。

## 实现

- `public_catalogs.py`：读取gz/zip SDF、CSV/TSV/parquet，重算质量、检查结构/键/公式，
  保留原始图、正式电荷、来源分类与SHA256；合并保留旧候选表示并联合来源。
  不剥离盐、不静默中和。源目录保留有电荷的连接结构；当前中性M检索路径排除它们。
  同一来源有真实中性形式时优先选它作为结构键代表，带电原图和质量仍保存在来源记录；
  不凭空生成中性结构。追加到现有库时依然保留现有表示。
- `chemical_priors.py`：只匹配适用模式/加合物；中性丢失使用带电前体减碎片，
  对两项测量误差采用相加上界。同一规则每谱最多一次，重复谱取最大证据。
  缺失不扣分、不删除候选，弱峰/质量偏差降低有效融合权重；相同支持保留旧次序。
- `secondary_inference.py`：显式 `chemical_priors` 配置启用低置信度重排，
  可与已校验的候选扩库组合；强谱库分支保持历史排名。拒绝未经验证的学习路由/
  direct ranker组合。输出 `.chemistry.json` 和路由审计。
- `chemical_audit.py`：无需神经权重即可检查真实谱中的规则覆盖及质量候选兼容性。

```sh
.venv/bin/python -m casmi_ml.chemical_audit \
  --queries data/test.parquet \
  --catalog external/chemical_catalogs/structures.parquet \
  --dictionary configs/chemical_priors.json \
  --output artifacts/chemical_priors/evidence.json
```

在既有有效部署recipe中加入下列块，path相对该recipe目录解析：

```json
{
  "chemical_priors": {
    "path": "chemical_priors.json",
    "sha256": "填入实际文件的SHA256",
    "weight": 0.0,
    "ppm": 10.0,
    "absolute_tolerance": 0.002,
    "intensity_floor": 0.01
  }
}
```

weight=0为不改变排名的对照。权重及容差必须在开发集冻结；尚无经真实独立验证
选择的正权重，现有生产recipe没有修改。实际待输出候选的最终SMILES用于化学评分，
包含历史检索候选，不只评分扩充池。

若既有recipe使用PubChemLite，应先将其与新库合并，再把合并文件作为唯一
`candidate_expansion.path` 并更新文件SHA256；不能直接用新库覆盖旧外部资产。
COCONUT继续从既有独立路径加载。

```sh
.venv/bin/python -m casmi_ml.public_catalogs merge \
  external/pubchemlite/structures.parquet \
  external/chebi/structures.parquet external/lipidmaps/structures.parquet \
  --output external/combined/structures.parquet
```

该命令需要还原此前PubChemLite资产。本轮已生成的新库联合目录为
`external/chemical_catalogs/structures.parquet`，仅包含ChEBI与LIPID MAPS。

## 后续必须做的四组实验

| 组 | 候选池 | 化学规则 |
|---|---|---|
| A | 原候选池 | 关闭 |
| B | 扩充候选池 | 关闭 |
| C | 原候选池 | 启用 |
| D | 扩充候选池 | 启用 |

固定同一编码器、谱图预处理、质量窗口、历史保护和已知/未知查询谱。
只在开发分子比较权重0、0.05、0.1、0.2、0.4，分别报告天然产物/timsTOF与其他来源；
用独立分子验收最终冻结方案。保留来源隔离及结构分组，不能从答案追加候选。
正式验收还需按官方RDKit2026.03.3互变异构体规范化后比较InChIKey14，
避免把多个等价表示计作不同候选。源目录保持原图，这一步属于评测/输出规范化。

每组报告候选Recall、候选内MRR、整体MRR@25、Top1/5/25、规则命中率、
首位改变及已知/未知分支、运行时间。未知MRR配对bootstrap95%差值下限>0，
已知MRR下降不超过0.001、Top1下降不超过0.005；通过后才建议更新部署。
对于混合比例未知的比赛，分场景收益比一个任意等权平均更有解释力。

当前缺少原始比赛训练谱、COCONUT完整候选资产和60K模型权重，尚未运行上述
真实消融；测试与数据库解析成功不能作为榜单提升证据。

## 资源与许可

- ChEBI release255（2026-09-09）：
  https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi.sdf.gz ，CC BY 4.0。
  https://www.ebi.ac.uk/chebi/about
- LIPID MAPS LMSD（2026-10-01）：
  https://www.lipidmaps.org/databases/lmsd/download ，SDF CC BY 4.0。
  实际下载 https://www.lipidmaps.org/files/?ext=sdf.zip&file=LMSD 。
- MS-FINDER表：
  https://github.com/systemsomicslab/MsdialWorkbench/tree/master/src/MSFINDER/MsfinderCommonStandard/Resources
  数据专属许可待核实；不套用软件许可或assembly许可。
- PubChem：
  https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest ，按开发数据预测的候选分子式
  预构建公开离线子集，推理不联网。`fastformula` 查询可用，但尚未执行。

截图中的数据库条数并非本轮数据统计；以下载资产的manifest为准。

## 本轮真实构建统计

| 导入产物 | 独立结构键 | 中性代表 | 带电代表 |
|---|---:|---:|---:|
| ChEBI | 134,451 | 130,814 | 3,637 |
| LIPID MAPS | 43,572 | 43,153 | 419 |
| 新库联合目录 | 162,175 | 158,416 | 3,759 |

联合目录不是“新增到生产库的命中数”，也不是所有数据库的完整原始条数。
源数据校验及键去重会减少条数；与COCONUT/PubChemLite还有未量化的重叠。
跨来源已有真实中性结构的118个键选用了中性表示，各原始图/质量均保存在来源记录。
仅合并新公共目录时可用 `merge --prefer-source-neutral`；该选项要求可验证的公共
导入manifest，默认合并仍保留现有部署候选的表示和质量。
