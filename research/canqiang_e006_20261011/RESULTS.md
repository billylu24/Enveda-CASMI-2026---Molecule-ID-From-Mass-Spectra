# E006 官方结果

本次提交 **57067224**，状态 **COMPLETE**，public MRR@25 **0.440**。

- 原始来源：Canqiang E006 公开 V1，30 个代码单元保持一致。
- 复现 notebook：[Kaggle V1](https://www.kaggle.com/code/giaok246/casmi26-canqiang-e006-reproduction)。
- 处理：窗口 40、ICE fallback=True，全部原模型和数据版本保留。
- 预览：12 个处理分子的完整排序与上游一致，400 行 CSV 字节 SHA256 一致；388 个占位行未直接提交。
- 比赛：通过同一 notebook V1 的 code submission 运行完整隐藏测试，完整处理断言通过后返回成绩。

验收见 `evidence/platform_acceptance.json` 和 `evidence/source_preview_comparison.json`；官方返回见 `evidence/submission_result.json`。可见预览不是独立验证，官方 public 成绩也不等于 private 成绩。
