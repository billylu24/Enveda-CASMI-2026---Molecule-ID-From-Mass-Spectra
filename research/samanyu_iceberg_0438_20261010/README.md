# Samanyu ICEBERG / GLACIER 复现

原始 notebook：https://www.kaggle.com/code/samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438，公开 V1。2026-10-10 获取源码及 16 个依赖版本，保留原作者和其上游作者署名。

本次使用原始 T4 镜像和模型。主进程 RDKit 2026.03.3，ICEBERG/GLACIER 隔离进程 RDKit 2025.03.6。全部原代码单元保持一致，保留原始权重、阈值、top-80 融合及 GLACIER 无可用分数时的 ICEBERG 回退排序，仅增加执行验收单元。

保存 notebook 时保持原 12 分子预览，正式比赛通过 **code submission** 运行完整隐藏测试及原预算。预览 CSV 的 388 行占位符不直接作为预测上传。平台验收核对 12 分子候选完整排序与上游公开 preview 一致，并验证模型、两个引擎、PubChem 和两个模拟器均执行成功。新增断言在正式 rerun 检查全部分子，拒绝任何未处理行。

`prepare_release.py` 校验全部原代码单元一致并生成复现 notebook。`publish.py` 固定发布版本并记录提交意图，避免重复提交。`monitor.py` 核对远程代码，下载预览执行证据，验收后提交同一版本、读取官方成绩并调用 `finish_git.py` 推送本次源码和结果。

上游标题 0.438 是来源声明；本次成绩以 Kaggle 实际返回为准。公开测试集与训练数据重叠，预览只验证执行与排序，不作为泛化评估。方法见 [METHODS_ZH.md](METHODS_ZH.md)。

复现流程：

```bash
.venv/bin/python research/samanyu_iceberg_0438_20261010/prepare_release.py
.venv/bin/python research/samanyu_iceberg_0438_20261010/publish.py push
.venv/bin/python research/samanyu_iceberg_0438_20261010/monitor.py
```

<!-- official-result -->
实际提交 `57038936`，状态 `COMPLETE`，public MRR@25 **0.438**。详见 [RESULTS.md](RESULTS.md)。
