# Canqiang E006 复现

原始 notebook：https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true，公开 V1。2026-10-11 获取源码、上游预览输出及 16 个数据依赖版本，保留原作者和所有上游署名。

本次完整保留 30 个原始代码单元，包括 E004 观察器和 E006 最终处理。使用原 Python 3.12 T4 镜像，主进程 RDKit 2026.03.3，模拟器隔离进程 RDKit 2025.03.6。仅新增来源说明和最终只读验收单元。数据版本冻结于发布前，全部最新更新早于原 notebook 的运行时间。

E006 在同一轮模型结果上，以官方 metric identity 为单位，对两个引擎进行 reciprocal-rank fusion，候选窗口 **40**。有 GLACIER 分数时保留原同分子式 GLACIER 排序；GLACIER 字典为空且 ICEBERG 字典非空时执行 **ICEBERG fallback**。保留 `safe_extend` 和原 E004 对照重建断言。条件是原始字典的 truthiness，不改为有限分数检测，也不改为之前 Samanyu 的 80 候选窗口。

保存 notebook 时按原开关只处理 12 个分子，400 行预览输出中另外 388 行为占位符。它们不作为预测 CSV 直接上传。正式比赛使用 **code submission**，Kaggle 在隐藏测试完整执行同一 notebook 版本。预览与上游 CSV 的 SHA256 和全部排序一致后才提交；隐藏重跑要求处理全部测试分子。

上游公开预览中 E006 相对 E004 改变 0 行：1 个分子进入 ICE 回退分支，但没有有限 ICE 分数。它仅证明预览可复现，不能证明隐藏测试无变化，也不是独立泛化评估。原页面 V1 显示 public score **0.440**，该成绩仅作为上游对照；实际成绩以本次 Kaggle 返回为准。源码中的 E004/其他作者历史分数不属于本次结果。

运行：

```bash
.venv/bin/python research/canqiang_e006_20261011/prepare_release.py
.venv/bin/python research/canqiang_e006_20261011/publish.py push
.venv/bin/python research/canqiang_e006_20261011/monitor.py
```

`publish.py` 记录版本及提交意图，阻止盲目重复提交。`monitor.py` 验收平台预览，提交同一版本，查询官方结果，生成报告并推送本次文件到 GitHub。`check_live.py` 只读查询运行状态。大型原始数据和模型保留在 Kaggle；GitHub 保存代码、依赖版本、执行验收和结果。

<!-- official-result -->
本次提交 **57067224**，状态 **PENDING**，完整隐藏测试正在运行，尚无官方成绩。 详见 [RESULTS.md](RESULTS.md)。
