# Offline chemical-prior hybrid release

This release extends the historical `hybrid.py`/COCONUT method with neutral
ChEBI and LIPID MAPS structures and five project-authored chemical rules.
Library confidence at least 0.5 preserves the historical output. The chemical
weight is fixed at 0.1. No neural checkpoint is required and no GAN is trained.

The input bundle was uploaded as a private dataset:
[CASMI 2026 ChEBI LMSD Chemistry Inputs](https://www.kaggle.com/datasets/xiaoyuzhoux120/casmi-2026-chebi-lmsd-chemistry-inputs).
The notebook is
[CASMI 2026 Chemical Priors Hybrid](https://www.kaggle.com/code/xiaoyuzhoux120/casmi-2026-chemical-priors-hybrid).
`status.json` records the verified run and submission state; preparing or running
a notebook alone does not mean that a competition submission has been made.

## Verified result — 2026-10-02

Notebook version 1 (`scriptVersionId=354585479`) completed CPU/offline execution
in 6m 57s. The competition accepted and reran this version on its hidden test
set. Its submission status is **Succeeded**, with displayed public MRR@25
**0.176**, matching the displayed historical best. No improvement is established.
See [the verified status](status.json) and
[the downloaded-output aggregate summary](validation_summary.json).

The catalog adds 77,572 structures beyond the COCONUT pool, after excluding
3,759 charged entries. All 400 visible example molecules used the protected
historical branch, so neither expansion nor chemistry changed those outputs.

| Fixed 32-example behavior control | MRR@25 | Recall@25 | Changed queries |
|---|---:|---:|---:|
| A: original pool, no rules | 0.138849 | 0.6875 | 0 |
| B: expanded pool, no rules | 0.136245 | 0.6875 | 7 |
| C: original pool, rules | 0.138849 | 0.6875 | 0 |
| D: expanded pool, rules | 0.136245 | 0.6875 | 7 |

Of these examples, 23 remained protected and only nine were eligible for the
new path. No rule matched the eligible queries. Candidate expansion slightly
reduced this diagnostic MRR; the small, previously observed sample does not
establish generalization. The rule mechanism still needs real-spectrum
coverage checks and an independent, molecule-disjoint ablation before any
benefit can be claimed. These results did not change the fixed weight.

The initial interactive attempt exposed a duplicate-column conversion error
in the full public catalog. The projection fix passed the actual 162,175-row
catalog check; the committed notebook then completed successfully. Final local
verification: 72 tests passed, one data-dependent test skipped, ten subtests.

## Rebuild and run

`build_notebook.py` embeds the current source files in a deterministic archive.
It needs the locally prepared `bundle/SHA256SUMS.json`; binaries are excluded
from Git. Regenerate only after reconstructing that input bundle from the
source manifests described in [the chemistry guide](../docs/CHEMICAL_PRIORS_20261001.md).

```bash
.venv/bin/python kaggle_release_chemistry/build_notebook.py
```

Import `notebook/chemical_priors_hybrid.ipynb` into Kaggle and attach the
competition, the private chemical input dataset, and
`aidensong123/casmi26-coconut-202609`. Run on CPU with internet disabled.
The notebook checks the frozen input checksums, installs the official
RDKit 2026.03.3 wheel offline, and produces `/kaggle/working/submission.csv`.
Save and run a version, then submit that version's CSV through the competition.

## Outputs and interpretation

- `submission.csv`: molecule IDs and up to 25 candidate SMILES.
- `submission.csv.report.json`: guard, catalog and chemistry usage counts.
- `submission.csv.routing.csv` and `.evidence.json`: per-query audit.
- `behavior_check.json`: fixed 32-molecule, masked-reference A/B/C/D controls.

The four controls independently switch candidate expansion and chemical
reranking. All selected keys are excluded from the reference spectra.
The examples have been observed before, so this is a behavior check rather
than independent validation. Its results do not select the weight or establish
leaderboard improvement. The public score of a new submission must be checked
separately. MS-FINDER resource tables with unverified data-specific licensing
are not included in the input bundle.
