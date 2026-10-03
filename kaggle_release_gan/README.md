# Conditional fingerprint GAN + shared graph candidates — 2026-10-02

This release tests whether conditional adversarial training improves spectrum-to-fingerprint ranking, using a matched supervised generator control. It also adds a bounded, formula-preserving graph-edit candidate generator shared by both neural arms. The GAN predicts fingerprints; it does **not** decode a molecular graph or SMILES.

## Formal run

- Kaggle notebook: [CASMI 2026 Chemical Priors Hybrid, version 2](https://www.kaggle.com/code/xiaoyuzhoux120/casmi-2026-chemical-priors-hybrid?scriptVersionId=354698716).
- Saved version name: **Conditional GAN Shared Graph Edits v2**.
- Frozen source commit: `a45c65e9f16501c554dfd4a12679003d630cb0ff`.
- Embedded source SHA-256: `ee7ccca6b373156e81522dea9598ccf4aeae1fe4eae1c4f827bc4383b4a2f93d`.
- GPU T4 x2, internet disabled; the runtime confirmed CUDA availability.
- Competition submission and measured results are tracked in [status.json](status.json). A notebook script version ID identifies a saved run, not a competition submission.

The version completed in 20,252.3 seconds and its `submission.csv` was actually submitted on October 2 (America/Los_Angeles). The hidden-test competition execution **Succeeded**, with public MRR@25 **0.171** verified on 2026-10-03, 0.005 below chemical-only v1 / historical best 0.176. [Scored submission proof](kaggle_submission_succeeded.jpg) and [matching version/source details](kaggle_submission_details_succeeded.jpg) are retained. The UI does not expose a competition submission ID. This entire-version decrease is not a GAN-only causal comparison. [Run proof](kaggle_run_success.png), [accepted submission proof](kaggle_submission_running.png), and [actual validation summary](validation_summary.json) are retained. Both selected model binaries match epoch 6 metadata; GAN selected weights contain 1,808 discriminator updates. Aggregate GAN increments over the matched supervised control have intervals crossing zero in both acceptance tasks. The complete archive, all retained case identities/ranks, all paired intervals and the 400-row visible CSV were independently rechecked. Formal training, output validation, competition submission and the public score are separate completion checks. The synthetic [cold smoke](cold_smoke_summary.json) verifies execution and checkpoint reload only; it is not evidence of chemical accuracy.

## Frozen comparison

[protocol.json](protocol.json) was fixed before training. Official canonical molecular identities separate training, development and acceptance. The intended caps are 60,000 training molecules, 1,000 development molecules and 1,000 acceptance molecules, at most two spectra per identity. Natural-product examples and detectable visible-query copies are excluded from neural fitting. Actual counts and exclusions are recorded by the runtime.

The generator has the same initialization, architecture, batches and training noise in both arms. Both have 12 epochs, two supervised warm-up epochs and a 1,800-second training ceiling per arm. The GAN adds alternating discriminator updates and an adversarial weight of 0.05. Checkpoints are selected by development supervised BCE; a GAN checkpoint requires actual discriminator updates. Acceptance does not select a checkpoint, route or submission. If the time ceiling changes actual update counts, the comparison must report that limitation.

Unknown-spectrum acceptance removes held-out identities and their canonical aliases from reference spectra and the library-derived pool. Independently sourced public structures remain legitimate candidates. Known-spectrum acceptance uses the same query cohort where references remain after all query-vector copies have been removed. These are proxy tasks, not the hidden Kaggle class labels.

The ranking report compares historical retrieval, expanded chemistry, raw supervised/GAN ranking and routed supervised/GAN ranking. Both neural arms receive exactly the same graph candidates. Routing preserves historical retrieval at confidence 0.75 and otherwise retains its first candidate, with fixed neural RRF weight 0.35 and chemical weight 0.1. One frozen acceptance evaluation records MRR@25, Top1/5/25, candidate recall, covered-candidate MRR, scaffold strata and paired molecule bootstrap intervals.

Graph edits swap two single-bond endpoints, reject invalid or disconnected products, preserve exact formula/mass and reject identities already in the eligible query pool. At most 32 new graphs are admitted per low-confidence query. Their novelty is local to that pool; global PubChem novelty and natural-product plausibility are not established. Graph counts summed across queries are not global unique-molecule counts.

## Reproduce

Attach the competition dataset, [COCONUT September 2026](https://www.kaggle.com/datasets/aidensong123/casmi26-coconut-202609) and the owner's [offline ChEBI/LMSD inputs](https://www.kaggle.com/datasets/xiaoyuzhoux120/casmi-2026-chebi-lmsd-chemistry-inputs). The last input contains the catalog, five project-authored chemical rules and an official PyPI RDKit Linux wheel; its manifest and all seven files are verified before execution. MS-FINDER reference tables are not redistributed because their data-specific license is unverified.

Import [the notebook](notebook/conditional_fingerprint_gan.ipynb), select a GPU and **Save & Run All** with internet disabled. The notebook embeds the recorded source archive and invokes `casmi_ml.adversarial_release.run`. It retrains from the competition input rather than requiring a previously unpublished model bundle. The notebook must rerun against the hidden competition test when submitted.

`build_notebook.py` rebuilds the embedded sources and [build manifest](build_manifest.json). Local verification passed 106 tests, one data-dependent skip and ten subtests; the cold start exercised preparation, both training arms, checkpoint serialization/reload, ranking, submission generation and cache cleanup.

The formal output keeps `submission.csv`, both checkpoints, training histories, `gan_report.json`, development and acceptance aggregate rankings, case audits, preprocessing and the prepared manifest. Large feature caches and copied raw spectra are removed after successful completion. Model binaries, full case rankings and raw competition data remain excluded from Git. Reports should distinguish public leaderboard performance, independent acceptance and the previously inspected historical examples.

See [the Chinese analysis and measured results](../docs/OVERNIGHT_RESULTS_20261002.md), [the failure audit](../docs/FAILURE_AUDIT_20261002.md), and [the generalization review](../docs/GENERALIZATION_REVIEW_20261002.md).
