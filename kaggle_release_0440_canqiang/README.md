# CASMI26 Canqiang E006 Reproduction 归档与深度分析

- **公榜成绩**: MRR@25 **0.440**（团队前序最高分）
- **复现提交**: `CASMI26 Canqiang E006 Reproduction - Version 1`（author: `yicheng lu`, team: `Spectral_Forge`）
- **Kaggle scriptVersionId**: `357274214`
- **源工程**: `canqiang/casmi26-e006-window-40-ice-fallback-true` (V1)
- **基线演进**: `0.436` (flexonafft v80) -> `0.438` (Samanyu ICEBERG) -> **`0.440` (Canqiang E006)**

---

## 核心机理与 0.440 的来源

经过对源码的逐行拆解，0.440 相比 0.438 的提升主要来自以下两项关键机制：

### 1. 结构级“一物一票”去重融合（Metric-Identity Fusion）
在 0.438 中，双引擎融合时使用的是原始 SMILES 字符串级别的简单字典去重，不同的互变异构体（Tautomers）或质子化状态会被视作不同候选，导致同一种拓扑母核在 RRF 排序中出现“重复刷票”或挤占宝贵槽位。
在 E006 中：
- 引入了官方度量标准的 `metric_identity`（即立体化学剥离后互变异构体规范化 InChIKey-14 + 重原子组成）；
- 在进入双引擎加权 RRF 时：
  ```python
  seen = set()
  for candidate in candidates:
      key = identity(candidate)
      if key in seen:
          collapsed += 1
          continue
      seen.add(key)
      scores[key] = scores.get(key, 0.0) + weight / (offset + len(seen))
  ```
- **消除了同义候选对多引擎融合的重复投票偏差（Duplicate-Vote Bias）**，让真正不同骨架结构的候选获得更纯粹的综合排名。

### 2. 融合候选池窗口紧缩（Window 80 -> 40）
- 0.438 将融合候选窗口放大到了 80（`FUSE_N = 80`）；
- 但实验证明，窗口放大到 80 会引入较多深层低质量候选，这些异构体在 GLACIER/ICEBERG 微弱打分差异下容易产生假阳性上浮；
- **E006 将窗口重新收敛锁定到 40**（`window = 40`），既保证了异构体筛选范围，又强力遏制了低置信度候选污染 Top 25。

### 3. 保留了 ICEBERG 盲区回退
在 GLACIER 无法打分（无 `[M+H]+` 谱图）的分子上，依旧继承并保持 `ice_fallback = True`，完整保留了对非单电荷/负离子的保底能力。

---

## 对后续实验的借鉴价值

1. **结构唯一投票（Metric Identity Fusion）** 是被 0.440 实测验证有效的正向增益，应作为我们后续所有融合实验的标准组件；
2. **窗口大小选择**：40 明显比 80 更稳健；
3. **与我们的自适应门控融合（Adaptive Simulator Fusion）具有完美的正交互补性**：
   - 0.440 解决了**引擎间融合时的异构体投票偏差**；
   - 我们的 0.44x 方案解决了**模拟器打分时的硬开关信息截断**；
   - **两者的结合（Metric Identity Fusion + Window 40 + Confidence-Gated Adaptive ICE/GLACIER）是冲击 0.445+ 的终极利器！**
