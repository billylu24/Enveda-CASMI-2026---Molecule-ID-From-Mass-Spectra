# Enveda CASMI 2026 — Molecule ID From Mass Spectra

This is a mass-filtered spectral-library retrieval baseline for the [Enveda CASMI 2026 competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

## Current results snapshot

Best public MRR@25: **0.176**. The chemical-prior historical-hybrid experiment was submitted successfully from notebook version 1 on 2026-10-02 and scored **0.176**, tying the historical hybrid. The preceding PubChemLite experiment scored **0.171**. See [the consolidated results and repository contents](docs/GITHUB_SNAPSHOT.md) for all submissions, validation caveats, and which assets must be regenerated.

## Chemical priors and generative-model analysis (2026-10-02)

Chemical-prior matching and ChEBI/LIPID MAPS/PubChem offline catalog import are
available as an experimental extension with a submitted public score of 0.176.
Independent chemical validation remains outstanding. See
[the chemistry integration guide](docs/CHEMICAL_PRIORS_20261001.md) for source
provenance, conservative motif rules, inference switches, and the required
four-way validation before deployment.

The [generative-model analysis](docs/GENERATIVE_MODELS_20261001.md) reviews
the spectrum-to-structure architecture. A real conditional fingerprint GAN,
a matched supervised generator control, canonical identity deduplication and
shared formula-preserving graph candidates are now implemented. Formal offline
GPU training is running as notebook version 2; its completion, submission and
score are tracked separately in [the GAN release](kaggle_release_gan/README.md).
The previous chemical-prior score of 0.176 does not establish GAN effectiveness.

The [Chinese analysis](docs/OVERNIGHT_RESULTS_20261002.md) compares our submission
history and leaderboard position, gives actual ranking failures and separates
candidate coverage, overfitting and limited extrapolation. The [GAN interpretation
checklist](docs/GAN_RESULT_INTERPRETATION_20261002.md) fixes how to interpret the
matched comparison and its uncertainty. The full engineering suite passed
106 tests, with one data-dependent skip and ten subtests passed.

The offline, checkpoint-free historical-hybrid extension completed the formal
Kaggle CPU run with internet disabled in **6m 57s** and its competition submission
status is **Succeeded**. It is documented in
[the Kaggle chemical release](kaggle_release_chemistry/README.md), with
ChEBI/LMSD candidates, a 0.5 confidence guard and a fixed 0.1 chemical weight.
[Its status file](kaggle_release_chemistry/status.json) records the run and
competition submission, and [the validation summary](kaggle_release_chemistry/validation_summary.json)
records the measured checks.

All 400 visible test molecules took the protected branch and their rankings
were unchanged. In the 32-key masked-reference behavior check, 23 were protected,
9 had low confidence, and no chemical rules matched. Original-catalog MRR@25
was 0.138849; adding the public catalog gave 0.136245, with or without the rules.
These previously observed examples are **not independent validation**. The
local regression suite passed 72 tests, with 1 skipped and 10 subtests passed.

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
