# Flexonafft「Closing the Evidence Gap」V30 复现

2026-10-17（美国洛杉矶时间）获取原链接的公开 **V30**。其源码内部配方名为 **V22 — BIO Structure Union**，配方名称与 Kaggle 发布版本号不同。原始 6 个代码单元和所有嵌入代码保持不变；只增加署名 Markdown 与末尾只读验收。

- 原作者：[Igor Zharov / flexonafft](https://www.kaggle.com/code/flexonafft/casmi26-closing-the-evidence-gap)
- 复现 Notebook：[giaok246 / V30 BIO Union Reproduction](https://www.kaggle.com/code/giaok246/casmi26-flexonafft-v30-bio-union-reproduction)（私有）
- 源码快照：[source/](source/)，来源与哈希：[source_snapshot.json](evidence/source_snapshot.json)
- 锁定依赖：[dependencies.json](evidence/dependencies.json)，发布身份：[release_identity.json](evidence/release_identity.json)
- 当前运行与验收结果：[RESULTS.md](RESULTS.md)

## 算法与运行环境

保留 sergiomalv BIO 结构优先级，补入 prvsiyan BIO 中按 engine2 source key 去重后尚未出现的结构。为新增结构重新计算 engine2 指纹与质量，不复用旧指纹数组。1200 秒准备预算超时会报错，不接纳不完整合并。其余 rel150 promotion、fragment-gap、fusion 与 metric-exact rank2 策略原样保留。完整继承署名见 [cell_01.py](source/cell_01.py) 与 [NOTICE](NOTICE)。

沿用来源 Docker 镜像、T4 GPU 与离线运行；主进程 RDKit 2026.03.3，模拟器隔离环境 RDKit 2025.03.6。16 个附件版本从来源 `lastRunTime` 之前的 Ready 版本选择；API 未暴露来源实际附件版本，因此这是明确记录的复现限制。Docker digest 与附件版本写入 [kernel-metadata.json](release/kernel-metadata.json)。

## 重跑与核验

在项目根目录，已有 Kaggle 登录且可访问比赛时：

```bash
.venv/bin/python research/flexonafft_evidence_gap_v30_20261017/prepare_release.py
.venv/bin/python -m unittest discover -s research/flexonafft_evidence_gap_v30_20261017 -p 'test_*.py' -v
.venv/bin/python research/flexonafft_evidence_gap_v30_20261017/publish.py push
.venv/bin/python research/flexonafft_evidence_gap_v30_20261017/live_log.py
.venv/bin/python research/flexonafft_evidence_gap_v30_20261017/publish.py status
.venv/bin/python research/flexonafft_evidence_gap_v30_20261017/fetch_evidence.py source
.venv/bin/python research/flexonafft_evidence_gap_v30_20261017/fetch_evidence.py run
.venv/bin/python research/flexonafft_evidence_gap_v30_20261017/verify_run.py
```

`push` 在已有发布意图时拒绝重复上传。迁移账户或建立另一次实验应先复制目录、修改 `prepare_release.py` 的 `KERNEL` 并建立独立状态；不要直接清除已有上传记录。`fetch_evidence.py` 只下载所选小文件，不记录签名下载 URL；版本变化时拒绝混合验收。`verify_run.py` 必须在平台 COMPLETE 且下载验收文件后运行。

5 项本地控制测试验证：原始源码/镜像一致性、合并优先级与新增指纹重算、不兼容指纹拒绝、超时拒绝不完整合并、metric twins 与 rank2 位置保护。这些不是准确率评估。宿主工具依赖见 [requirements-tools.txt](requirements-tools.txt)；推理环境由 Docker 与平台运行版本记录。

## 结果边界

来源普通运行仅处理 **12 个分子**，输出 400 行中的其余 **388 行是占位行**。本复现保留该行为和完整隐藏测试路径。占位行不算完成推理，不能直接把整个预览 CSV 作为有效比赛预测上传。

上游源码中的 **0.451** 是其引用的基础配方成绩声明，不是本次 BIO-union 复现成绩。`submission_v22_control.csv` 也是本次适配配方的同轮输出，不是未改动基础 V22 的独立对照。本次任务上传 Notebook 与 GitHub，不自动创建新的比赛评分提交。

V15 的历史复现保留在 [../flexonafft_evidence_gap_20261014/](../flexonafft_evidence_gap_20261014/)，其官方提交 57217289 已查证完成，public MRR@25 为 **0.441**；这不是 V30 的成绩。
