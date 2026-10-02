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
