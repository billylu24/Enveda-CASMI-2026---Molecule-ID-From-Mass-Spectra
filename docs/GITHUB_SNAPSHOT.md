# Results snapshot — 2026-09-29

## Kaggle public results

| Submission | Method | Public MRR@25 |
|---|---|---:|
| 56533793 | Spectral library baseline | 0.140 |
| 56534532 | Multi-spectrum consensus | 0.141 |
| 56535378 | Historical hybrid + COCONUT | **0.176** |
| 56606429 | Guarded MLP | 0.162 |
| 56667967 | Historical/neural routing | 0.169 |
| 56688097 | 60K residual neutral-loss model | 0.173 |
| 56690108 | Experimental PubChemLite expansion | 0.171 |

Scores were checked using the Kaggle CLI on 2026-09-29. These are public leaderboard scores, not private leaderboard results. Local holdouts differ between experiments; compare each experiment to its paired baseline, not across cohorts.

The 10.7M-parameter residual model passed its independent local acceptance gate. The direct graph ranker failed independent validation. The calibrated router was withheld because of transfer risk; the robust router failed development protection gates. PubChemLite improved independent local point estimates, but its confidence interval crossed zero and its statistical acceptance flag remains false. Its public score did not improve on the preceding release. Historical hybrid remains the best public result.

## Repository contents

Source, tests, configs, experiment guides, compact aggregate reports, training histories, deployment recipes, and Kaggle notebook sources are included. Selected reports retain their original `artifacts/` paths so documentation references remain useful. `results/published_files.json` lists the curated result assets and their SHA-256 hashes. `results/kaggle_submissions.csv` records the score check.

Competition raw data, model checkpoints, full candidate rankings, per-molecule training/evaluation records, binary caches, virtual environments, and packaged datasets are excluded. This is a source-and-results snapshot, not a self-contained pretrained inference package. Model checksums in recipes identify excluded assets; the recipes cannot run until those assets are restored or regenerated. Existing reports may contain original local paths and timing metadata for provenance.

## Reproduction

1. Create the Python 3.12 environment and download competition data as described in the root README (Kaggle access and competition acceptance required).
2. Install `requirements-ml.txt` and an appropriate official PyTorch build for neural experiments; see `docs/EXPERIMENTS.md`. The local GPU environment is not shipped.
3. Obtain COCONUT and, for expansion, PubChemLite from the sources in `external/COCONUT_ATTRIBUTION.md` and `external/pubchemlite/ATTRIBUTION.md`. Both use CC BY 4.0; source manifests and attribution are retained, not the datasets.
4. Follow the dated experiment guides to rebuild caches, train checkpoints, and reproduce inference. Training requires substantially more resources than running tests. Numerical/library differences may change near-tied rankings.
5. Run tests: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v`.

Kaggle notebook sources and release statuses are retained under `kaggle_notebook/` and `kaggle_release*/`. Attached private bundle datasets may require owner access; public notebook links do not imply that model bundles are publicly downloadable. Historical statuses and protocols are preserved, including failed experiments.
