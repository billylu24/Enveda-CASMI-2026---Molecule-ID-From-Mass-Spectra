# 复现范围与方法

复现目标是用户指定的 `samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438`。冻结的是 2026-10-10 Kaggle API 返回的 public V1；同时保存 pull 源码、API metadata、SHA256、执行镜像和 16 个带版本的依赖。标题 0.438 不作为本次复现成绩。

第二引擎先在独立子进程中产生 top-40 候选。PubChem 通道随后在独立子进程中搜索、评分并输出候选及 popularity 信息。主引擎结合光谱库、FPNet 和 LightGBM 排名，再进行原始 pool popularity 调整。保持原 notebook 的权重、阈值、候选上限和生成配置。

ICEBERG 与 GLACIER 在隔离的 RDKit 2025.03.6 环境里预测谱图。主进程固定 RDKit 2026.03.3。GLACIER 使用 `[M+H]+` 谱图；双引擎按原始 RRF 参数融合至 80 个候选，GLACIER 没有有效分数的分子按原代码启用 ICEBERG 回退。保留原 PubChem 提升规则、门控插槽和所有相同分子式组内排序。

原始最终 tail 单元以自身给出的 metric identity 去重，保留每个有效身份首次出现的顺序，补入融合溢出、主引擎尾部、第二引擎及 PubChem 候选。在原附件提供 `dreams_np_*` 时才使用其质量匹配候选；不补造缺失数据。保留原单调排名检查。

全部原代码单元保持一致，包括 12 分子公开预览和完整 hidden rerun 开关。新增末尾验收单元检查 active molecule IDs、主引擎错误、第二引擎与 PubChem 完整性、模拟器执行状态和支持覆盖、有效 SMILES、候选上限及 metric identity 唯一性。正式 hidden rerun 要求输出中的全部 IDs 都已处理，不能使用预览占位行。

平台 notebook 完成后，核对远程代码与固定发布源码一致，下载并校验输出 SHA256，比较上游公开 preview 的全部 12 分子候选排序。只有验收通过才提交该固定 notebook 版本；正式评分运行完整隐藏输入，绝不直接提交含 388 行占位符的 preview CSV。提交意图及响应均持久化，若响应不明确则不自动重试；评分轮询本身只读。GitHub 推送只包含本次复现源码、可复用脚本和摘要证据；原工作区中的其他未提交文件保留。

公开 test 与 train 重叠。可见预览运行是执行验证，不能据其得分报告模型泛化能力；报告的 public MRR@25 只来自这一次官方 competition submission 返回。
