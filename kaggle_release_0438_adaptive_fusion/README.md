# CASMI26 Adaptive Simulator Fusion (0.438 -> 0.44x 升级版)

- **基准公榜**: MRR@25 **0.438** (`CASMI26 Samanyu ICEBERG 0.438 Reproduction`, scriptVersionId `357007287`)
- **核心升级目标**: 突破 `0.438` 瓶颈，解决原版二值硬开关 (`CELL19_ICE = False`) 带来的信息截断问题。

## 升级背景与理论依据

在原版 0.438（v14）中，作者发现：
1. **全量均等融合（`lam_ice=1.0, lam_gl=1.0`）**：公榜大幅下跌至 `0.402`（ICE 噪声干扰了 GLACIER）；
2. **纯 GLACIER 关停 ICE**：公榜为 `0.417`；
3. **二值硬开关（有 GLACIER 则 ICE 设为 0，无 GLACIER 则退回 ICE=1.0）**：公榜达 `0.438`。

然而，对于存在 `[M+H]+` 质谱的分子，GLACIER 内部对同分异构体的区分度并不总处于高置信区间。当 GLACIER 的打分 margin 极小（几乎持平）时，机械地将 ICEBERG 设为 0 导致丢失了有效的谱图比对证据。

## 本次升级内容（Cell 20 置信度感知自适应融合）

1. **同分异构体区分度估计 (`_gl_margin`)**:
   在候选池 InChIKey 空间中，度量 GLACIER 最高两名得分差距 $\Delta_{\text{GL}}$；
2. **三态置信度路由**:
   - **态 A (GLACIER 盲区, 无 `[M+H]+`)**: 完全由 ICEBERG 兜底重排 (`lam_ice=1.0`, 精确对齐 0.438 表现)；
   - **态 B (GLACIER 高置信, $\Delta_{\text{GL}} \ge 0.03$)**: 纯信 GLACIER (`lam_ice=0.0`, 严格保护已有优势)；
   - **态 C (GLACIER 低置信/震荡, $\Delta_{\text{GL}} < 0.03$)**: 激活温和互补融合 (`lam_ice = 0.20`, `lam_gl = 1.00`)，以小权重借助 ICEBERG 消除假阳性，杜绝 1.0 大权重导致的噪声反噬。
3. **完全兼容前序流程与最终度量安全审计**:
   不改变任何上游模型推理，不增加额外推理时长（依旧保持 ~16 分钟 T4x2 完成）。

## 文件清单与 SHA256

- `casmi26-adaptive-simulator-fusion.ipynb`: 可直接上传至 Kaggle 执行的 Notebook
- `notebook.py`: 逐 cell 导出的 Python 源码
- 详见 `SHA256SUMS`
