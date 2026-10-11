# CASMI26 Samanyu ICEBERG 0.438 Reproduction 归档

- **公榜成绩**: MRR@25 **0.438**（当前团队最高纪录，排名第 193 位，铜牌区间）
- **复现提交**: `CASMI26 Samanyu ICEBERG 0.438 Reproduction - Version 2`（author: `yicheng lu`, team: `Spectral_Forge`）
- **Kaggle scriptVersionId**: `357007287`
- **源工程**: `samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438` (V1)
- **环境**: Pinned GPU image, RDKit 2026.03.3, GPU T4 x2
- **运行时长**: 16m 51s (1010.8s)

## 核心架构流程

1. **Engine 2 (Subprocess)**: Two-ranker engine (BIO + AFIX)，在干净 GPU 显存下运行，产出 Top 40 候选及 InChIKey-14；
2. **PubChem 流行度信道**: 搜索 ~105M PubChem 候选，结合指纹匹配与流行度先验（`POP_LAM = 0.25`, `POP_UNION = 200`）；
3. **Engine 1 (v4n)**: Ahmed Berat Özer 的 160 特征 4-booster LightGBM ranker，结合结构池流行度重排（`mu = 0.15`）；
4. **ICEBERG + GLACIER 质谱模拟器**: 对分子式相同的异构体进行预测谱图比对重排；
5. **门控 PubChem 合并**: 当库内最佳匹配置信度低时介入；
6. **双引擎加权 RRF 融合**: $\alpha = 0.6$, $K_{rr} = 3.0$，保留 Top 80 窗口；
7. **GLACIER 盲区回退**: 针对无 `[M+H]+` 质谱的分子，退回 ICEBERG 单模重排；
8. **度量安全尾部审计**: 保持已有答案单调性，去重并补齐至 25 个候选。

## 文件校验

详情见 `SHA256SUMS` 与 `verified_result.json`。
