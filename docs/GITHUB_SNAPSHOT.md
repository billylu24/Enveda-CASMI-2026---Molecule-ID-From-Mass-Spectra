# Results snapshot — 2026-10-02

## Kaggle public results

| Submission / notebook version | Method | Public MRR@25 |
|---|---|---:|
| 56533793 | Spectral library baseline | 0.140 |
| 56534532 | Multi-spectrum consensus | 0.141 |
| 56535378 | Historical hybrid + COCONUT | **0.176** |
| 56606429 | Guarded MLP | 0.162 |
| 56667967 | Historical/neural routing | 0.169 |
| 56688097 | 60K residual neutral-loss model | 0.173 |
| 56690108 | Experimental PubChemLite expansion | 0.171 |
| Notebook version 1 | [Chemical-prior historical hybrid](https://www.kaggle.com/code/xiaoyuzhoux120/casmi-2026-chemical-priors-hybrid) | **0.176** |

The seven historical scores were checked using the Kaggle CLI on 2026-09-29.
The chemical-prior experiment's competition status **Succeeded** and public
MRR@25 **0.176** were checked in Kaggle on 2026-10-02. Its notebook version is 1
(`scriptVersionId` 354585479); the UI did not expose a competition submission ID,
so none is recorded. These are public leaderboard scores, not private leaderboard
results. Local holdouts differ between experiments; compare each experiment to
its paired baseline, not across cohorts.

The 10.7M-parameter residual model passed its independent local acceptance gate. The direct graph ranker failed independent validation. The calibrated router was withheld because of transfer risk; the robust router failed development protection gates. PubChemLite improved independent local point estimates, but its confidence interval crossed zero and its statistical acceptance flag remains false. Its public score did not improve on the preceding release. The chemical-prior extension ties the historical hybrid's best reported public result; it does not improve that score.

### Chemical-prior submission and behavior checks

The formal notebook run completed on CPU with internet disabled in **6m 57s**.
Its guard threshold is 0.5 and its predetermined chemical weight is 0.1; no GAN
was trained. See [the release guide](../kaggle_release_chemistry/README.md),
[submission status](../kaggle_release_chemistry/status.json), and
[validation summary](../kaggle_release_chemistry/validation_summary.json).

The submitted input catalog contains 162,175 raw structure keys. The neutral-mass
path excluded 3,759 charged representatives, and 77,572 structures were appended
after removing overlap with COCONUT. All 400 visible test molecules were protected
by the historical confidence guard, with **zero changed rankings**.

The fixed 32-key, masked-reference behavior check had 23 protected and 9
low-confidence molecules. No chemical rules matched the nine eligible queries;
the rules were bypassed for the 23 protected molecules.

| Behavior arm | Public catalog expansion | Chemical rules | MRR@25 |
|---|---|---|---:|
| A | Off | Off | 0.1388493724842409 |
| B | On | Off | 0.13624520581757424 |
| C | Off | On | 0.1388493724842409 |
| D | On | On | 0.13624520581757424 |

Arm A had Top1 0 and Top25 0.6875. Expansion in arms B/D changed 7 rankings,
with 27 new mass candidates and 19 new output candidates in the audit totals.
These previously observed examples are **not independent validation**, and
the results were not used to tune the weight. The public-score tie and protected
visible outputs do not establish independent chemical effectiveness or improved
generalization. Independent chemical ablation remains outstanding. A new paired
GAN experiment is running; its results are recorded separately below.

### Conditional fingerprint GAN experiment

The real conditional generator/discriminator implementation and matched supervised
control were frozen at commit `a45c65e9f16501c554dfd4a12679003d630cb0ff`. Formal
Kaggle notebook version 2 (`scriptVersionId` 354698716), named **Conditional GAN
Shared Graph Edits v2**, completed on GPU T4 x2 with internet disabled in
20,252.3 seconds. Its `submission.csv` was actually submitted; the competition
status is **Notebook Running** during hidden-test execution, not Succeeded.
No new competition score is recorded yet.

Both neural arms share the candidate pool and bounded formula-preserving graph
edits. The GAN generates fingerprints, not molecular graphs; the graph edits are
locally novel candidates rather than established globally new molecules. See
[the release](../kaggle_release_gan/README.md), [frozen protocol](../kaggle_release_gan/protocol.json),
[live status](../kaggle_release_gan/status.json), [Chinese analysis](OVERNIGHT_RESULTS_20261002.md)
and [result interpretation](GAN_RESULT_INTERPRETATION_20261002.md). Engineering
verification passed 106 tests, one skip and ten subtests; a real synthetic cold
start exercised training, checkpoint reload and complete inference. These checks
are not chemical generalization results. The actual formal aggregate report and
both selected checkpoint binaries are now reviewed: unknown routed MRR is
0.016693 supervised / 0.016840 GAN, known MRR 0.782591 / 0.781980. Both
GAN-minus-supervised paired 95% intervals cross zero; graph-only new truth
coverage is zero. Full case identity/bootstrap recomputation remains pending.
See [validation summary](../kaggle_release_gan/validation_summary.json). The [runtime audit](RUNTIME_REVIEW_20261002.md)
documents repeated CPU structure work and constraints for a later inference-only release.

## Repository contents

The October 1 update adds offline public-catalog import, strict MS-FINDER
dictionary adaptation, five literature-supported chemical-prior rules,
optional low-confidence reranking, evidence audits, and regression tests.
See [the chemical-prior guide](CHEMICAL_PRIORS_20261001.md) and
[the architecture and GAN analysis](GENERATIVE_MODELS_20261001.md).
The additions remain experimental: the submitted chemical-prior experiment ties
0.176, while its independent chemical ablation remains outstanding. The October 2
update adds the paired conditional fingerprint GAN and shared graph candidates,
official identity split/deduplication and detailed failure audits. Local verification
passed 106 tests, with one data-dependent test skipped and ten subtests passed.

Downloaded ChEBI/LIPID MAPS data and derived catalogs remain excluded. The
MS-FINDER tables are also excluded because their data-specific license has
not been verified. Only the adapters and project-authored rules are published.

Expanded catalogs carrying a `formal_charge` column exclude nonzero or unknown
charges from the neutral-mass inference path, independently of the chemistry
switch. When merging legacy catalogs with charge-aware sources, missing charges
are computed from the retained SMILES without changing their structure or mass.

Source, tests, configs, experiment guides, compact aggregate reports, training histories, deployment recipes, and Kaggle notebook sources are included. Selected reports retain their original `artifacts/` paths so documentation references remain useful. `results/published_files.json` lists the curated historical result assets and their SHA-256 hashes. `results/kaggle_submissions.csv` records the September 29 CLI score check; the October 2 submission is recorded in `kaggle_release_chemistry/status.json` and this snapshot.

Competition raw data, model checkpoints, full candidate rankings, per-molecule training/evaluation records, binary caches, virtual environments, and packaged datasets are excluded. This is a source-and-results snapshot, not a self-contained pretrained inference package. Model checksums in recipes identify excluded assets; the recipes cannot run until those assets are restored or regenerated. Existing reports may contain original local paths and timing metadata for provenance.

## Reproduction

1. Create the Python 3.12 environment and download competition data as described in the root README (Kaggle access and competition acceptance required).
2. Install `requirements-ml.txt` and an appropriate official PyTorch build for neural experiments; see `docs/EXPERIMENTS.md`. The local GPU environment is not shipped.
3. Obtain COCONUT and, for expansion, PubChemLite from the sources in `external/COCONUT_ATTRIBUTION.md` and `external/pubchemlite/ATTRIBUTION.md`. Both use CC BY 4.0; source manifests and attribution are retained, not the datasets.
4. Follow the dated experiment guides to rebuild caches, train checkpoints, and reproduce inference. Training requires substantially more resources than running tests. Numerical/library differences may change near-tied rankings.
5. Run tests: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v`.

Kaggle notebook sources and release statuses are retained under `kaggle_notebook/` and `kaggle_release*/`. Attached private bundle datasets may require owner access; public notebook links do not imply that model bundles are publicly downloadable. Historical statuses and protocols are preserved, including failed experiments.
