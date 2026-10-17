# V30 BIO Structure Union 复现结果

Kaggle 私有 Notebook **V1** 已 **COMPLETE**，运行验收通过。原始 V30 的 **6 个代码单元及嵌入代码完全保留**。12/12 个实际预览分子完成推理，388 行占位预测排除在推理验收外。

**首选 12/12、候选集合 12/12、完整排序 12/12 与上游一致**。完整预览 CSV 与上游逐字节一致。

| 检查项 | 本次结果 |
|---|---:|
| 原始代码单元完全保留 | 6/6 |
| 真实处理分子 / CSV 行数 | 12 / 400 |
| 占位行 | 388 |
| 基础引擎错误 | 0 |
| 首选与上游一致 | 12/12 |
| 候选集合与上游一致 | 12/12 |
| 完整排名与上游一致 | 12/12 |
| BIO 新库 / 旧库结构数 | 98,433 / 62,744 |
| 补入 / 重叠 / 无效结构 | 69 / 62,675 / 0 |

原代码计时约 **20.6 分钟**。主进程 RDKit 2026.3.3；ICEBERG 与 GLACIER 都使用 CUDA 和隔离的 RDKit 2025.03.6，健康检查通过。BIO 合并计数与上游相同，所有新增结构沿用 engine2 原生指纹重算。

本次最终 CSV SHA256：`383812e3be50974cd6f2f2959fe94a25869fa68318b248aff1454341c1cd1992`。

上游 V30 最终 CSV SHA256：`383812e3be50974cd6f2f2959fe94a25869fa68318b248aff1454341c1cd1992`。

## 结果边界

本次验证了普通运行的 **12 分子预览路径**。400 行 CSV 中有 388 行占位，不能直接把该预览 CSV 当作有效比赛预测提交。完整隐藏测试路径保留，但本任务只上传 Notebook 与 GitHub，**没有创建新的 V30 比赛评分提交，因此没有本次官方 MRR@25**。

源码内提到的 **0.451** 是继承的基础配方声明；`submission_v22_control.csv` 也是适配配方的同轮输出，不能当成未修改的 0.451 基础 V22 对照。预览一致性不证明独立准确率或隐藏集泛化收益。历史 V15 的官方提交 **57217289** 已完成，成绩 **0.441**，独立记录在旧目录中。

5 项本地控制测试全部通过：原始源码一致性、BIO 优先级/去重/重算、不兼容指纹拒绝、超时拒绝部分合并、rank2 的 metric twin 与首选保护。测试输入是合成控制，不是准确率评估。

## 可核查证据

- [发布身份与源码哈希](evidence/release_identity.json)
- [16 个冻结数据依赖](evidence/dependencies.json)；原附件版本不可从 API 获取，选择来源运行前的 Ready 版本并保留该限制
- [Kaggle 上传回执与远端身份](evidence/publication_state.json)
- [平台验收与逐分子/逐位比对](evidence/platform_acceptance.json)
- [实际运行验收与包版本](evidence/platform/reproduction_acceptance.json)
- [实际运行日志](evidence/platform/runtime_log.json)
- [BIO 合并审计](evidence/platform/eng/bio_union_audit.json)
- [本次真实处理预测](evidence/platform/processed_predictions.csv)；[上游真实处理预测](evidence/source_reference/processed_predictions.csv)
