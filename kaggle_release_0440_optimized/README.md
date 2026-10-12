# CASMI26 0.440 架构工程重构与加速优化版 (E011 Runtime-Optimized)

- **基准公榜**: MRR@25 **0.440** (Canqiang E006 Reproduction, scriptVersionId `357274214`)
- **核心优化目标**: 
  1. 100% 保持 0.440 的“Metric-Identity 结构级一物一票 + Window=40 + ICE Fallback”底层逻辑；
  2. 针对全量 400 分子执行 2 小时+ 的运行时瓶颈，实施**置信度早停剪枝**与**计算预算动态分配**；
  3. 保障全量隐藏推理在 40~50 分钟内高效完成，为后续加合物解耦与天然产物先验迭代提供稳定的工程底座。

## 关键工程优化

1. **显著性置信度早停（Significance Early-Exit）**:
   - 若检索阶段 Top 1 分数与 Top 2 分数差距显著且拥有高质量特征匹配（高确信度样本），直接锁定 Top 1 结果，避免高代价的神经模拟器冗余前向推理；
2. **模拟器动态预算（Adaptive Time-Budget Allocator）**:
   - 对 GLACIER / ICEBERG 实施逐分子动态预算与早停，限制长尾异常耗时分子，消除阻塞；
3. **严格格式与 0.440 等价性守护**:
   - 输出保持官方 InChIKey-14 规范化去重，严格通过 E004 观察者与 E006 阶乘验收断言。

## 文件清单与 SHA256

- `casmi26-canqiang-e006-optimized.ipynb`: 优化加速后的 Kaggle Notebook
- `notebook.py`: 逐 cell 导出的 Python 脚本
- `SHA256SUMS`: 完整校验和
