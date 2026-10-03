"""Training-only observed-chemistry/motif association against pinned near-mass controls."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from casmi_ml.chemistry import RULES
from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import DIM, KINDS, evidence_features
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.training_query_pairs import paired_rows


def association(strength, positive, negative):
    strength = np.asarray(strength)
    positive = np.asarray(positive, dtype=bool)
    negative = np.asarray(negative, dtype=bool)
    if strength.shape != positive.shape or negative.shape != (len(positive), 15):
        raise ValueError(
            "Training strengths, positive motif and pinned15 motif controls must align"
        )
    counts = []
    for lower, upper in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.00001)]:
        use = (strength > lower) & (strength <= upper)
        n = int(use.sum())
        pos = float(positive[use].mean()) if n else None
        neg = float(negative[use].mean()) if n else None
        informative = use & (positive.astype(np.int8)[:, None] != negative).any(1)
        counts.append(
            {
                "lower_exclusive": lower,
                "upper_inclusive": min(upper, 1.0),
                "training_molecules": n,
                "positive_motif_fraction": pos,
                "negative_motif_fraction": neg,
                "positive_minus_negative_fraction": pos - neg if n else None,
                "training_molecules_with_discriminative_pinned_negative": int(
                    informative.sum()
                ),
            }
        )
    return counts


def run(output):
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "feature_sha256": digest("casmi_ml/diagnostic_feature_runtime.py"),
            "chemistry_sha256": digest("casmi_ml/chemistry.py"),
            "training_sha256": digest(TRAIN),
            "negative_table_sha256": digest(HARD),
            "selection": "All original60K keys; same RNG42 representative and RNG260 same-ion partner; max precise chemical evidence across pair",
            "comparison": "Broad rule SMARTS in positive vs same pinned15 training-only near-mass structures. Fixed strength bands (.0,.1],(.1,.3],(.3,.6],(.6,1], no dev labels/keys fit or selection.",
            "local_only": True,
            "independent_acceptance": False,
            "scope": "Descriptive training association, not causal ion proof, model fitting, or full-system evidence",
        },
    )
    started = time.monotonic()
    train = pd.read_parquet(TRAIN)
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    groups = train.groupby("inchikey14", sort=True).indices
    development = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    if (
        list(groups) != list(data["keys"])
        or set(groups) & development
        or len(groups) != 60000
    ):
        raise ValueError("Training/dev key isolation failed")
    rng = np.random.default_rng(42)
    ids = np.array([rng.choice(v) for v in groups.values()])
    paired = paired_rows(train, ids)
    positions = np.column_stack([ids, paired]).reshape(-1)
    patterns = [Chem.MolFromSmarts(r.smarts) for r in RULES]
    if any(pattern is None for pattern in patterns):
        raise ValueError("Rule SMARTS invalid")
    support = []
    for smiles in train.iloc[ids].normalized_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid original training positive")
        support.append([molecule.HasSubstructMatch(p) for p in patterns])
    support = np.asarray(support, dtype=bool)
    chemical = (
        np.stack(
            [
                evidence_features(r, True)
                for r in train.iloc[positions].to_dict("records")
            ]
        )
        .reshape(60000, 2, DIM)
        .max(1)
    )
    associations = {}
    for i, rule in enumerate(RULES):
        for j, kind in enumerate(KINDS):
            strengths = chemical[:, i * 3 + j]
            associations[rule.name + ":" + kind] = {
                "baseline_training_positive_motif_fraction": float(
                    support[:, i].mean()
                ),
                "nonzero_evidence_molecules": int((strengths > 0).sum()),
                "strength_bands": association(
                    strengths, support[:, i], support[hard, i]
                ),
            }
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "training_rows": len(positions),
        "train_development_overlap": 0,
        "associations": associations,
        "seconds": time.monotonic() - started,
        "scope": "Broad SMARTS association against pinned near-mass training negatives; positive associations are not mechanistic proof, confidence calibration or an accuracy gain",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
