# CASMI26 Adaptive Metric-Identity Simulator Fusion (0.440 -> 0.445+ 升级版)

- **基准公榜**: MRR@25 **0.440** (`CASMI26 Canqiang E006 Reproduction`, scriptVersionId `357274214`)
- **核心升级目标**: 深度融合 0.440 的“一物一票去重（Metric-Identity Fusion）+ 窗口 40”优势，叠加置信度感知自适应模拟器重排（Confidence-Gated Adaptive Simulator Fusion）。

## 核心融合设计

1. **结构唯一投票（Metric-Identity Fusion）**:
   彻底消除异构体对多引擎 RRF 的重复投票偏差（Duplicate-Vote Bias）；
2. **紧凑重排窗口（Window = 40）**:
   过滤 40~80 窗口的深层低质候选，强力抑制模拟器在尾部假阳性候选上的虚高漂移；
3. **GLACIER 置信度自适应门控**:
   - **GLACIER 明确置信（Margin $\ge 0.03$）**: 100% 信任 GLACIER（$\lambda_{\text{ICE}} = 0.0, \lambda_{\text{GL}} = 1.0$）；
   - **GLACIER 犹豫胶着（Margin $< 0.03$）**: 激活温和互补融合（$\lambda_{\text{ICE}} = 0.20, \lambda_{\text{GL}} = 1.0$）；
   - **GLACIER 盲区（无 `[M+H]+`）**: 100% 由 ICEBERG 兜底（$\lambda_{\text{ICE}} = 1.0, \lambda_{\text{GL}} = 0.0$）。

## 运行保护

- E004 对照重建固定使用原始纯 GLACIER 路径；自适应 ICEBERG 融合只在 E006 实验分支启用。
- 可见运行仍是 12 分子的 smoke check；只有隐藏重跑才会处理完整 400 个分子。
- 提交前必须看到运行成功，并确认 `processed=400`、`preview_placeholders=0` 与 `same_run_control_reconstructed=true`。

## 文件清单与 SHA256

- `casmi26-adaptive-metric-fusion.ipynb`: 可直接上传至 Kaggle 执行的升级 Notebook
- `notebook.py`: 逐 cell 导出的 Python 脚本
- 详见 `SHA256SUMS`
