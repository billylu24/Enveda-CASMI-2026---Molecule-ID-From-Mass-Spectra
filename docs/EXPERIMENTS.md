# CPU spectrum-to-fingerprint experiments

## 2026-09-28 follow-up ablations

The latest guarded submission scored 0.162, below historical hybrid's 0.176.
The follow-up experiment is isolated from the original frozen artifacts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -u -m casmi_ml.ablation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u -m casmi_ml.visible_ablation
```

Results are in `artifacts/ablation_20260928/` and `docs/ABLATION_20260928.md`.
The first command compares COCONUT-only versus combined-library analog candidates,
15 versus 35 ppm windows, the previous production guard, and 24 routing settings
using the original training-only checkpoint. It audits charge-aware mass conversion
and per-spectrum mass unions separately as candidate-recall experiments.
The development winner is frozen before sampling a previously unused 2,000-molecule
holdout. A second, unknown-spectrum-prioritized endpoint was separately frozen
before any fresh predictions were completed; see `secondary_selection.json`.
Both endpoints retain their identity regardless of holdout results.

Known and unknown conditions share one molecule cohort. Known evaluation removes
exact query spectra and their copies; it reports only molecules with remaining
reference spectra. The equal-weight objective is a declared proxy because competition
class proportions are unknown. Historical candidate-rule controls share the prepared
reference index and omit historical nearest-reference fallback; the visible reproduction
uses the complete original reference-loading and fallback code instead. This distinction
must remain explicit when interpreting local scores. Model rankings use supplied 2D keys;
the visible reproduction additionally checks actual SMILES-derived identities.

The scripts do not change the production recipe, retrain models, or submit to Kaggle.
Completed cached results are reused; use a new experiment directory and protocol for
further tuning after these fresh labels have been examined.

## Reproduce

Run from the repository root using Python 3.12. Original baseline commands remain
unchanged. Neural experiments need PyTorch in addition to `requirements.txt`:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u -m casmi_ml.experiment run
.venv/bin/python -m casmi_ml.experiment predict --output submission_neural.csv
.venv/bin/python -m casmi_ml.experiment package --output kaggle_bundle
```

The `run` command prepares data if missing, benchmarks all encoders, trains five
controlled experiments, repeats two finalists with seeds 43/44, freezes one
selection on development data, evaluates holdout and diagnostic splits, and
refits the selected encoder(s) on train+development. It does not contact Kaggle,
upload datasets, or submit predictions. Use the root argument for an independent
experiment. Never reuse a holdout to tune a new configuration.

Individual stages are available:

```bash
.venv/bin/python -m casmi_ml.experiment prepare --root artifacts/new_experiment
.venv/bin/python -m casmi_ml.experiment benchmark
.venv/bin/python -m casmi_ml.experiment train --architecture mlp --seed 42 --seconds 7200
.venv/bin/python -m casmi_ml.experiment evaluate --split dev
```

`train --config` reads epochs and learning rate from the configuration; architecture
and seed are explicit flags. `run --config` reads the whole experiment schedule.
All stage outputs are under `artifacts/experiment` by default. Completed runs are
reused. A new configuration requires a new root, rather than overwriting a frozen
selection. An interrupted run without `result.json` restarts that run; completed
checkpoints remain on disk. New jobs reserve their full time slot before launch;
a hard-killed job keeps that reservation, conservatively preventing budget reuse.

## Validation contract

Splits use `SHA256("42:" + inchikey14)[:8] % 10000`: 0–7999 train,
8000–8999 development, and 9000–9999 holdout. All molecules appearing in
`enveda-np-examples` are assigned to a separate diagnostic set, regardless of
other sources. Reservoir sampling selects at most four spectra per selected
molecule using a fixed RNG. Diagnostic queries use the named source only.

The manifest records sampling counts and source size. The catalog records each
molecule's split; sampled parquet files record original row IDs. Categorical
vocabularies and numeric normalization are fitted on sampled training data only.

Reference spectra for validation belong only to training molecules, including
training molecules outside the 20,000 neural training sample. The candidate pool
contains training-split structures plus the external COCONUT database. Held-out
labels are not appended. Thus the correct structure is often absent: report
candidate recall and conditional MRR alongside overall MRR. This is an unseen
structure retrieval proxy, not the original known-spectrum validation and not a
Kaggle leaderboard estimate.

The diagnostic set here also excludes all same-molecule reference spectra and
uses at most four query spectra per molecule. It must not be compared directly
to the historical 0.90 known-spectrum diagnostic results.

All encoders use the same candidate pool (35 ppm with a 0.006 Da floor), fingerprint
target (Morgan radius 2, 2048 bits), BCE objective, and candidate likelihood score.
The common-pool retrieval control uses the existing 0.65/0.35 consensus and
protected-head mixing rule, but supplies all common-pool structures to analog
ranking. It is a controlled adaptation of `hybrid.py`, not a bitwise reproduction
of the historic 15 ppm COCONUT candidate pipeline. E0's new score is only comparable
to experiments within this protocol.

## Models and budgets

- `mlp`: 1 Da spectrum bins → 512 → 256 → 2048 fingerprint logits.
- `enhanced`: spectrum bins + precursor-minus-fragment bins + metadata, same MLP.
- `metadata`: enhanced model without the mass-difference histogram (ablation).
- `deepsets`: shared continuous-peak MLP 19 → 64 → 128, masked mean/max pooling.
- `transformer`: same peak embeddings, two 128-wide, four-head encoder layers,
  256-wide feedforward blocks, masked mean pooling.

Peak encoders use the 64 strongest peaks. The 19 features contain normalized m/z,
square-root relative intensity, normalized precursor mass difference, and sine/cosine
features at four fixed frequencies for mass and mass difference. There is no
sequence-position embedding. Empty spectra have one valid zero token; padding is
masked. Metadata comprises precursor m/z, collision energy mean/min/max, missingness,
and one-hot adduct, ionization mode and instrument. Unknown categories have their
own position. Instrument collision-energy units are not inferred from raw text.

The schedule is in `configs/experiments.json`. Wall time for benchmarking and
training jobs (including per-epoch development evaluation) is budgeted; preparation,
reference indexing and independent final evaluation are separate. Each encoder
gets at most 20 epochs with patience three. The training budget is a ceiling, not
a requirement to consume 24 hours. The common dataset is reduced only if the
slowest measured epoch would exceed 30 minutes. The final refit uses the union of
the sampled train and development sets (up to 22,000 molecules in this run), which
is below the 50,000-molecule cap. Two selected encoders share the final three-hour
allowance. Parameters and process peak RSS are recorded; RSS includes preparation
and evaluation data retained by the process.

## Selection and delivery

Two architecture finalists are selected on development MRR; configurations within
0.005 prefer shorter run time and fewer parameters. Across seeds, the strongest
mean selects the single-encoder candidate. Compare common-pool retrieval alone,
one neural model, retrieval+one model and retrieval+two models using RRF constant
60 and neural weights 0/0.25/0.5/0.75/1. Prefer fewer/smaller encoders within 0.005.
The selection is frozen before holdout evaluation. Bootstrap intervals resample
molecules. A fixed deployment accept/reject gate retains retrieval if the frozen
neural candidate does not exceed retrieval on holdout; it never searches other
models or weights using holdout results. Final-refit weights have not independently been re-evaluated on holdout.

`final_selection.json` is the sole inference selection. Checkpoints contain model
configuration and preprocessing, rather than depending on test-fitted state.
Final inference restores all allowed competition library spectra. It averages
fingerprint probabilities across a molecule's spectra, scores mass-matched
structures, fuses ranks, and falls back to nearest library hits if needed.
Submission validation checks molecule IDs, counts, parseability, and 2D uniqueness.

The package contains a notebook, Python sources, selected weights, runtime version
record, report and COCONUT attribution. Competition and COCONUT data are attached
separately, not copied into the package. The inference notebook expects CPU-capable PyTorch in the
Kaggle environment; the bundle includes the exact training-version RDKit wheel for CPython 3.12 / Linux x86_64. The
notebook fails clearly if required mounted resources are absent or ambiguous.

Current Kaggle rules and runtime limits were **not** reverified because browser
access to Kaggle was denied. The package follows the repository's prior offline
submission convention. It is not an uploaded dataset or a verified Kaggle run.

To rebuild the bundled RDKit wheel resource using the official Python registry:

```bash
uvx --from pip pip download --only-binary=:all: --no-deps rdkit==2026.3.6 --dest external/ml_wheels --python-version 312 --implementation cp --abi cp312 --platform manylinux_2_28_x86_64
```

Input file SHA256 checksums are recorded in `input_checksums.json`; packaged code,
weights and resources have a separate `SHA256SUMS.json`. The notebook installs the
bundled RDKit version offline when the environment version differs, so candidate
fingerprints use the same RDKit release as training.

## Compatibility failure and confidence routing

The initial global neural fusion is additionally checked on the original NP source:
queries from `enveda-np-examples`, references from all other sources, with no query
labels appended to the candidate pool. A decrease exceeding 0.005 MRR fails this
acceptance check. This is a binary gate, not a search on that diagnostic set.

If global fusion fails, `guard` selects a confidence threshold from
0.5/0.65/0.75/0.85 **on development only**. For confident spectral hits it keeps the
complete retrieval ranking. Otherwise it protects the retrieval first candidate
and fills the remaining positions from the frozen neural RRF ranking. It uses the
same encoder and fusion weight already chosen in the first experiment.

Development selection maximizes unseen-spectrum MRR subject to known-spectrum MRR
being within 0.005 of retrieval. The known development condition allows other spectra
of the query molecule, but removes the query peaks and identical spectrum copies
from the reference library. Only structures actually present in those remaining
reference spectra are added to COCONUT candidates.

After freezing the threshold, a **new 2,000-molecule holdout cohort**, disjoint from
all previously sampled molecules, evaluates both conditions. Acceptance requires
an improvement in unseen-spectrum MRR and no more than 0.005 deterioration in the
known-spectrum condition. Bootstrap intervals are reported; these are local proxy
metrics, not proof of a public leaderboard improvement. Failed acceptance retains
the historically submitted `hybrid.py`. Original selections and results are kept.

```bash
.venv/bin/python -m casmi_ml.experiment audit-known
.venv/bin/python -m casmi_ml.experiment guard
```

The complete `run` command includes these stages. A rejected global neural recipe
is retained as `research_selection.json`; packaging preserves its weights alongside
the accepted production selection. The final notebook only uses `final_selection.json`.
