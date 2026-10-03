# Enveda CASMI 2026 — Molecule ID From Mass Spectra

This is a mass-filtered spectral-library retrieval baseline for the [Enveda CASMI 2026 competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

## Current results snapshot

Best public MRR@25: **0.176** (historical hybrid). The latest PubChemLite experiment scored **0.171**. See [the consolidated results and repository contents](docs/GITHUB_SNAPSHOT.md) for all submissions, validation caveats, and which assets must be regenerated.

## Local setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -f train.parquet -p data
.venv/bin/kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -f test.parquet -p data
.venv/bin/python baseline.py --data-dir data --output submission.csv
```

The Kaggle code submission is generated from the same `baseline.py` source in `kaggle_notebook/casmi_baseline.ipynb`. The script uses the `train.parquet` and current `test.parquet` attached to the competition, with no network access required during execution.

A candidate is retained if its neutral molecular formula mass is within 35 ppm of a test precursor's inferred neutral mass. Fragment peaks are binned at 0.1 Da, weighted by square-root intensity and m/z, then compared with cosine similarity. For each molecule, the maximum similarity over its spectra ranks unique 2D structures.

This baseline retrieves known structures from the provided library. It does not generate novel structures de novo, so its ceiling is limited on the competition's novel-molecule class.

## First submission (2026-09-24)

- Kaggle notebook: https://www.kaggle.com/code/giaok246/casmi-2026-mass-filtered-spectrum-search-baseline
- Submission ID: `56533793` (notebook version 2)
- Public MRR@25: **0.140**; rank at check: **1128 / 1452** teams.
- Local format check: 400 distinct molecule IDs, no missing predictions, 3–25 candidates per molecule. The top-ranked SMILES matched between the local and Kaggle runs for all 400 example molecules.

## Improved models (2026-09-24)

The validation in `improved.py` treats all `enveda-np-examples` spectra as queries, excludes that source from the search library, and scores by `inchikey14`. It covers 250 natural-product structures and 1,184 spectra. The deterministic odd/even key halves give:

| Ranking | Half 1 MRR@25 | Half 2 MRR@25 |
|---|---:|---:|
| Original best single spectrum | 0.873 | 0.863 |
| Multi-spectrum consensus | **0.900** | **0.917** |
| Fine peak and neutral-loss reranking | 0.896 | 0.886 |

`consensus.py` implements the selected multi-spectrum ranking. Kaggle notebook version 3 scored **0.141** (submission `56534532`).

`coconut_experiment.py` simulates a structure with no reference spectrum by masking its `inchikey14` from all library candidates. It adds the public [COCONUT September 2026 structures](https://www.kaggle.com/datasets/aidensong123/casmi26-coconut-202609) as mass-matched candidates and ranks them by molecular-fingerprint similarity to the best spectral neighbors. In this class-2 proxy, COCONUT candidates scored **0.177 / 0.197** MRR@25 on the two halves; library-only retrieval scored zero by construction. This validation covers known COCONUT structures, not truly novel molecules.

`hybrid.py` combines the two candidate lists. Kaggle notebook version 5 scored **0.176** (submission `56535378`), compared with **0.140** for the original baseline. Rank at check: **950 / 1453** teams. The notebook attaches the COCONUT dataset and a public [offline RDKit wheel](https://www.kaggle.com/datasets/dmitriigluzdov/rdkit-2025-09-6-cp312-manylinux-wheel); its internet access remains disabled.

To reproduce the latest local output:

```bash
.venv/bin/kaggle datasets download -d aidensong123/casmi26-coconut-202609 -f coconut_structures.parquet -p external
.venv/bin/python hybrid.py --data-dir data --coconut external/coconut_structures.parquet --output submission_v3.csv
```

COCONUT structure data is CC BY 4.0; attribution details are in the upstream dataset's `COCONUT_ATTRIBUTION.md`.

## MLP → Transformer CPU experiments

The new `casmi_ml` pipeline trains fingerprint predictors with molecule-disjoint
validation, shared candidate pools and a bounded CPU schedule. It compares MLP,
feature-enhanced MLP, DeepSets and Transformer, selects a single inference recipe,
and can produce an offline Kaggle bundle. See [the experiment guide](docs/EXPERIMENTS.md)
for commands, validation limitations and packaging requirements.

```bash
.venv/bin/python -u -m casmi_ml.experiment run
.venv/bin/python -m casmi_ml.experiment predict --output submission_neural.csv
.venv/bin/python -m casmi_ml.experiment package --output kaggle_bundle
```

Measured results and the frozen recommendation are written to
`artifacts/experiment/REPORT.md`; no Kaggle submission is automated.

### Completed CPU experiment (2026-09-26)

The selected submission recipe uses **MLP + metadata with confidence-protected
spectral retrieval**: retain the reference ranking at consensus confidence ≥0.65;
otherwise preserve its first candidate and fill using neural RRF (weight 0.75).
On a fresh, disjoint 2,000-molecule holdout, unseen-spectrum MRR@25 improved from
0.00816 to 0.01008. On 1,146 molecules with other reference spectra, MRR changed
from 0.81790 to 0.81624, within the predefined 0.005 tolerance. These are local proxy
results, not Kaggle scores. The initial unprotected neural blend was rejected for
severely degrading known-spectrum retrieval.

See [measured results](docs/RESULTS.md). Deliverables are `kaggle_bundle.zip`,
`kaggle_bundle/casmi_winner.ipynb`, and `submission_neural.csv`. The packaged code
reproduced the 400-molecule CSV byte-for-byte from outside the project directory;
all visible test molecules used the protected retrieval branch. Kaggle notebook
[`giaok246/casmi-2026-guarded-mlp-fingerprint`](https://www.kaggle.com/code/giaok246/casmi-2026-guarded-mlp-fingerprint)
version 1 completed successfully on CPU with internet disabled on 2026-09-27.
Competition submission `56606429` scored **0.162**, checked on 2026-09-28, below the historical hybrid score **0.176**; see `kaggle_release/status.json`. The guarded recipe is therefore not the best public-scoring submission.

### Follow-up ablations (2026-09-28)

Completed 29 development comparisons and a new disjoint 2,000-molecule holdout.
The unknown-spectrum-prioritized frozen recipe improved local MRR from 0.01237
to 0.01628, with known-spectrum MRR 0.79390 → 0.79419 (1,132 molecules).
The previous guarded recipe remained stronger on the known-spectrum proxy but
failed the new unknown-spectrum improvement gate. These results do not establish
a new Kaggle score. Production configuration is unchanged.
See [the ablation report](docs/ABLATION_20260928.md) and
`artifacts/ablation_20260928/experimental_recipe.json`.

### Submitted historical/neural routing (2026-09-29)

Notebook [Historical Neural Routing](https://www.kaggle.com/code/giaok246/casmi-2026-historical-neural-routing) version 1 completed CPU/offline inference. Competition submission **56667967** scored **0.169**, above the previous guarded 0.162 but below historical hybrid 0.176. Its 400 visible rankings match the previous historical hybrid Kaggle output in full. Release files and status are in `kaggle_release_secondary/`; see [next experiments](docs/NEXT_STEPS_20260929.md).

### GPU model scaling (2026-09-29)

Trained five encoder architectures on 20K molecules and three finalists/controls on 60K molecules using an RTX 5070. The frozen 10.7M-parameter residual MLP with neutral-loss features passed independent CPU-float32 acceptance: unknown-spectrum MRR 0.01572 → 0.02021 (2,000 molecules), known-spectrum MRR 0.79143 → 0.79333 (1,213 molecules), against the previously submitted neural recipe. Two additional seeds supported the development improvement.

See [the scaling report](docs/SCALE_20260929.md). Candidate inference configuration is `artifacts/scale_20260929/deployment_recipe.json`; the prepared offline release is in `kaggle_release_scale/`. Submitted as **56688097**, notebook [Residual 60K Inference](https://www.kaggle.com/code/giaok246/casmi-2026-residual-60k-inference) version 2; public score **0.173**, above previous neural 0.169 but below historical 0.176.

### Calibrated routing (2026-09-29)

With the large encoder frozen, molecule-grouped development validation selected logistic H/F routing from 53 configurations. On a new 2,000-molecule holdout, unknown-spectrum MRR improved from 0.01883 to 0.02195 (+16.6%; paired difference CI95 [0.00108, 0.00546]). Known-spectrum MRR was 0.78674 → 0.79168 (1,171 molecules); its difference interval crosses zero. This passes the predeclared point-estimate protection gate, without demonstrating statistical non-inferiority for known spectra.

See [the router report](docs/ROUTER_20260929.md). Local inference configuration: `artifacts/router_20260929/deployment_recipe.json`. Evaluation and deployment share the same feature calculation; 50 real groups matched exactly. No new Kaggle submission has been made.

Full visible inference generated 400 valid rows, but the calibrated router changes 99 top-1 predictions despite near-unit reference similarities. Treat this as a transfer risk requiring calibration review before competition deployment; the frozen model was not retuned on these observations.

### Direct spectrum–structure ranking (2026-09-29)

Completed candidate-miss decomposition, rebuilt matched single-query validation, and trained fingerprint and graph candidate encoders with mass-neighbor listwise negatives on 60K molecules. The selected graph/old-neural blend improved development MRR but failed a new 2,000-molecule holdout: unknown MRR 0.01774 → 0.01666, known 0.71900 → 0.71846. It was not uploaded or submitted. Final acceptance used the exact per-group CPU deployment path; 27 tests passed. See [the direct-ranking report](docs/DIRECT_RANK_20260929.md).

### PubChemLite candidate expansion (2026-09-29)

Added an independently sourced, validated PubChemLite catalog (469,579 structures, CC BY 4.0). A new 2,000-molecule holdout showed unknown candidate recall 8.25% → 12.00% and MRR 0.01970 → 0.02232; known MRR 0.70862 → 0.70872. The unknown difference CI95 [-0.000333, 0.005788] crosses zero, so the predeclared statistical gate **failed**. Both point estimates improved; under the user's instruction to upload local improvements, this was released explicitly as an **experimental submission**, without changing the failed acceptance flag.

The offline package matched local output byte-for-byte, 25 real low-confidence rankings matched the deployment functions exactly, and 30 tests passed. [Report](docs/CANDIDATE_EXPANSION_20260929.md); [Kaggle notebook](https://www.kaggle.com/code/giaok246/casmi-2026-pubchemlite-expansion-inference); live release status: `kaggle_release_expansion/status.json`.

PubChemLite experimental submission **56690108** was submitted from notebook v1; public score **0.171**, below the 60K model (0.173) and historical hybrid (0.176). All 400 platform-visible rankings match the previous Kaggle output.

### Chemical evidence and generation research (2026-10-01)

Added precise diagnostic-ion and neutral-loss-combination evidence, offline
MetFrag reranking, matched fresh cohorts, three controlled peak-encoder training
methods (supervised/masked/DINO), and spectrum-conditioned SMILES generation with
predicted formula conditions. Each stage has frozen configuration, reproducible
checksums, and a shared bounded training ledger. The continuous loop now uses the explicitly authorized development gate for
experimental publication; independent acceptance is reported separately. See [the research guide](docs/RESEARCH_20261001.md) for commands,
assumptions, GAN feasibility, and the distinction between generation pilots and
full performance acceptance.

The accepted MetFrag chemistry model has an offline Kaggle package builder:
`.venv/bin/python -m casmi_ml.chemistry_release`. The notebook bundles Java 21
and the validated RDKit wheel, verifies checksums, and falls back to retrieval
when the fragmentation time budget expires. Submission 56769471 uses notebook
v4; current status is recorded in `kaggle_release_chemistry/status.json`.
See [the next-round improvement plan](docs/NEXT_IMPROVEMENTS_20261002.md) for
candidate coverage, mass-hypothesis, and generation-ranking priorities.

持续研究循环：参见 [运行与停止说明](docs/RESEARCH_LOOP.md)、[冻结配置](configs/research_loop.json) 和 [汇总研究结果](results/research_loop)。当前重复开发最佳第163轮：2,000未知分子MRR@25为0.041611，相对第149轮增加0.001283（+3.18%）；1,340已知分子MRR为0.604673，Top1保持0.565672。仅在原高置信新增尾部分支中，实际碎裂证据通过时将候选移到原有前3名之后，明确反证时移除新增尾部；缺失或预算不足保留第149轮。75项真实无标签完整排名重放和2000/1340排名指标重建通过；开发碎裂缓存经多轮补齐，不是单次冷部署证据。第163轮400行空缓存推理通过：1136.3秒/6058.6MiB，193个高置信新增组全部碎裂完成，无预算回退；平台验证待完成，尚未比赛提交。第149轮已通过平台400排名一致验证（1006.9秒/3210.8MiB），等待额度。最近第12/13/17/24轮公共分数为0.175/0.175/0.174/0.175，历史公共最佳0.176保留。每日额度下一次在UTC2026-10-04 00:00恢复，合格版本顺序发布，每次最多一个比赛提交待评分。

后续对照：第151扩大高置信预筛500无改善；第155碎裂支持完整MRR0.040489、第156排序融合0.040484、第161保护前3的碎裂前移0.040726，均未达到对149的发布门槛。质量条件训练背景校准和LOTUS目录初筛也未给出扩大依据。Java进程复用的12组fresh双臂真实碎裂分数全部相同，70.3秒→58.0秒（1.21倍）；第162轮以固定1200秒预算检查更多高置信碎裂评分，仍需完整门槛、无标签重放和新的冷资源/平台验证。所有持续选择属于重复开发，未新增独立验收；失败轮次和实现修复均保存并同步GitHub。
