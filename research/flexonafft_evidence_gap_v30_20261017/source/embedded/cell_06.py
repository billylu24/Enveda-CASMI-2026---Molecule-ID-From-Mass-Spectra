'''# CASMI26 V1.1 — metric audit and safe tail

Starting point: this account's V1 submission, public MRR **0.425** on 2026-10-08.
This is a conservative candidate-coverage experiment, not a claimed leaderboard improvement.

Changes: pin official RDKit 2026.03.3; preserve the V1 inference/ranking pipeline;
deduplicate only after ranking using the official metric identity; append ranked overflow
and mass-compatible DreaMS candidates into vacant slots; assert every original matchable
identity has equal or better rank. The old DreaMS branch was bypassed by `NameError: Path`.

No hidden-test labels, manual labels, or network access are used. The visible test overlaps
training data, so its score is an operational check and is **not** a validation estimate.

Source attribution: copied V1 ensemble from BobtheBot369 / Ahmed Berat Özer / prvsiyan
and the other authors credited in its code. Scoring reference: Kaggle Competition Metrics,
`https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank`, version 14, Apache-2.0.
'''
