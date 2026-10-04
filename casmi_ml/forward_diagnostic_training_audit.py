"""Train-only held-out audit of structure-to-measured diagnostic signatures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.candidate_chemistry_specificity_audit import CACHE, contrast_summary
from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import evidence_features
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.full_gap_candidate_compatibility import structural_support
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives
from casmi_ml.training import configure
from casmi_ml.training_query_pairs import paired_rows


def heldout_mask(keys):
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(("forwarddiag20261004:" + key).encode()).digest()[:8],
                "big",
            )
            % 5
            == 0
            for key in keys
        ],
        bool,
    )


def fit_ridge(features, targets, fit_rows, *, device="cuda"):
    """Fixed mean squared error plus .01 L2; unpenalized training intercept."""
    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    y = torch.as_tensor(targets, dtype=torch.float64, device=device)
    fit = torch.as_tensor(fit_rows, dtype=torch.bool, device=device)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or fit.shape != (len(x),):
        raise ValueError("Aligned molecular features/targets/train split required")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not fit.any():
        raise ValueError("Finite training data and nonempty fit split required")
    mean_x, mean_y = x[fit].mean(0), y[fit].mean(0)
    centered_x, centered_y = x[fit] - mean_x, y[fit] - mean_y
    n = int(fit.sum())
    covariance = centered_x.T @ centered_x / n
    covariance += 0.01 * torch.eye(x.shape[1], dtype=x.dtype, device=device)
    weights = torch.linalg.solve(covariance, centered_x.T @ centered_y / n)
    predicted = ((x - mean_x) @ weights + mean_y).clamp(0, 1)
    return predicted.cpu().numpy(), {
        "weights": weights.cpu().numpy(),
        "mean_x": mean_x.cpu().numpy(),
        "mean_y": mean_y.cpu().numpy(),
    }


def run(output):
    configure(42, threads=4)
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only single GPU training diagnostic required")
    output = Path(output)
    source_protocol = json.loads((CACHE / "protocol.json").read_text())
    cache_report = json.loads((CACHE / "report.json").read_text())
    for field, source in {
        "training_sha256": TRAIN,
        "negative_table_sha256": HARD,
        "feature_sha256": "casmi_ml/extended_diagnostic_rules.py",
        "chemistry_sha256": "casmi_ml/chemistry.py",
    }.items():
        if source_protocol[field] != digest(source):
            raise ValueError("Isolated measured training cache changed")
    for name, sha in cache_report["private_input_sha256"].items():
        if digest(CACHE / name) != sha:
            raise ValueError("Exact training cache asset required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "hard_table_sha256": digest(HARD),
            "cache_report_sha256": digest(CACHE / "report.json"),
            "rule": "Original60K/120K sameRNG42/RNG260 paired measured mean/sqrt60channels. Hash by molecule into fixed~80% fit/~20% internal holdout, all rows of each molecule same side. Compare fixed.01L2 ridge structure2048fp->60 measured diagnostic signatures against same20SMARTS->60 ridge and training-mean intercept. Equal1/60 positive-only observed-query dot predicted-candidate score, same pinned15 and mass mask. No development training/evaluation or threshold/regularizer selection; no absent-peak penalty, original model unchanged. This signature marginalizes original measured energies/adducts and is not a spectrum simulator.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    train = pd.read_parquet(TRAIN)
    groups = train.groupby("inchikey14", sort=True).indices
    devkeys = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    if list(groups) != data["keys"] or set(groups) & devkeys:
        raise ValueError("Original isolated training molecular identities required")
    rng = np.random.default_rng(42)
    ids = np.array([rng.choice(v) for v in groups.values()])
    pair = paired_rows(train, ids)
    rows = pd.read_parquet(CACHE / "training_rows.parquet")
    if (
        list(rows.inchikey14) != data["keys"]
        or not np.array_equal(rows.representative_row, ids)
        or not np.array_equal(rows.paired_row, pair)
    ):
        raise ValueError("Original measured training pair binding differs")
    positions = np.column_stack([ids, pair]).reshape(-1)
    original = []
    for start in range(0, len(positions), 1024):
        if c.stopped():
            return None
        original.extend(
            evidence_features(r, True)
            for r in train.iloc[positions[start : start + 1024]].to_dict("records")
        )
    targets = np.concatenate(
        [
            np.stack(original).reshape(60000, 2, 24).mean(1),
            np.load(CACHE / "mean_chemical.npy"),
        ],
        axis=1,
    )
    support = np.asarray(
        [structural_support(s) for s in train.iloc[ids].normalized_smiles], np.float32
    )
    holdout = heldout_mask(data["keys"])
    fit = ~holdout
    predicted = {"training_mean": np.broadcast_to(targets[fit].mean(0), targets.shape)}
    for name, features in [
        ("broad_motif_ridge", support),
        ("fingerprint_ridge", data["fps"]),
    ]:
        predicted[name], model = fit_ridge(features, targets, fit)
        np.savez_compressed(output / ("private_" + name + ".npz"), **model)
    compatible = admissible_negatives(data["masses"], hard)[holdout]
    heldout_competitors = holdout[hard[holdout]]
    query = targets[holdout]
    arms = {}
    for name, values in predicted.items():
        positive = (query * values[holdout]).mean(1)
        negative = (query[:, None, :] * values[hard[holdout]]).mean(-1)
        arms[name] = {
            "heldout_mse": float(np.square(values[holdout] - query).mean()),
            "channel_heldout_mse": np.square(values[holdout] - query).mean(0).tolist(),
            "positive_evidence_contrasts": contrast_summary(
                positive, negative, compatible
            ),
            "both_molecules_heldout_mass_contrasts": contrast_summary(
                positive, negative, compatible & heldout_competitors
            )["mass_compatible"],
        }
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": len(targets),
        "fit_molecules": int(fit.sum()),
        "internal_holdout_molecules": int(holdout.sum()),
        "train_development_overlap": 0,
        "internal_holdout_target_used_for_fit": False,
        "arms": arms,
        "scope": "Training-internal heldout diagnostic-signature prediction and fixed positive-only pair contrast. No new external acceptance cohort, dev metrics, calibrated cosine fusion, spectrum simulator, full system gain or model activation. A useful result only motivates separately bound matched capacity/system experiments.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
