"""Fixed training-internal molecule-heldout diagnostic prediction with singleton CE."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.data import category, write_json
from casmi_ml.dino_energy_pairs import energy_signature
from casmi_ml.direct_experiment import train_data
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.full_gap_candidate_compatibility import evidence_features
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze
from casmi_ml.training import configure


def feature_block(fps, energies, contexts, interaction=False):
    fps, energies, contexts = map(np.asarray, (fps, energies, contexts))
    if (
        fps.ndim != 2
        or contexts.ndim != 2
        or energies.shape != (len(fps),)
        or len(contexts) != len(fps)
    ):
        raise ValueError(
            "Aligned structure, singleton CE and categorical context required"
        )
    values = [fps, contexts, energies[:, None] / 100]
    if interaction:
        values.append(fps * (energies[:, None] / 100))
    result = np.concatenate(values, axis=1)
    if not np.isfinite(result).all() or (energies < 0).any():
        raise ValueError("Finite nonnegative measured CE required")
    return result


def fit_streaming_ridge(block, targets, fit, device="cuda", batch=512):
    """Same .01 L2/unpenalized intercept, training moments accumulated in batches."""
    targets, fit = np.asarray(targets), np.asarray(fit, dtype=bool)
    n = int(fit.sum())
    if (
        not n
        or targets.ndim != 2
        or len(targets) != len(fit)
        or not np.isfinite(targets).all()
    ):
        raise ValueError("Finite molecularly isolated training targets required")
    dimension = block(np.arange(min(1, len(fit)))).shape[1]
    covariance = torch.zeros((dimension, dimension), dtype=torch.float64, device=device)
    cross = torch.zeros(
        (dimension, targets.shape[1]), dtype=torch.float64, device=device
    )
    sum_x = torch.zeros(dimension, dtype=torch.float64, device=device)
    sum_y = torch.zeros(targets.shape[1], dtype=torch.float64, device=device)
    fit_ids = np.flatnonzero(fit)
    for start in range(0, n, batch):
        ids = fit_ids[start : start + batch]
        x = torch.as_tensor(block(ids), dtype=torch.float64, device=device)
        y = torch.as_tensor(targets[ids], dtype=torch.float64, device=device)
        sum_x += x.sum(0)
        sum_y += y.sum(0)
        covariance += x.T @ x
        cross += x.T @ y
    mean_x, mean_y = sum_x / n, sum_y / n
    covariance /= n
    covariance -= mean_x[:, None] * mean_x[None, :]
    covariance.diagonal().add_(0.01)
    cross /= n
    cross -= mean_x[:, None] * mean_y[None, :]
    weights = torch.linalg.solve(covariance, cross)
    predicted = np.empty(targets.shape, dtype=np.float64)
    for start in range(0, len(fit), batch):
        ids = np.arange(start, min(start + batch, len(fit)))
        x = torch.as_tensor(block(ids), dtype=torch.float64, device=device)
        predicted[ids] = ((x - mean_x) @ weights + mean_y).clamp(0, 1).cpu().numpy()
    return predicted, {
        name: value.cpu().numpy()
        for name, value in (
            ("weights", weights),
            ("mean_x", mean_x),
            ("mean_y", mean_y),
        )
    }


def run(output):
    configure(42, threads=4)
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU training diagnostic required")
    output = Path(output)
    started = time.monotonic()
    previous = Path(
        "artifacts/research_loop/rounds/0382_diagnostic_energy_stability_audit"
    )
    prior = json.loads((previous / "protocol.json").read_text())
    if prior["source_sha256"] != digest(
        "casmi_ml/diagnostic_energy_stability_audit.py"
    ) or prior["training_sha256"] != digest(TRAIN):
        raise ValueError("Source-bound complete measured energy audit required")
    if not (previous / "report.json").exists():
        raise ValueError("Complete measured energy evidence required")
    cache_report = json.loads((CACHE / "report.json").read_text())
    if (
        digest(CACHE / "training_rows.parquet")
        != cache_report["private_input_sha256"]["training_rows.parquet"]
        or cache_report["train_development_overlap"]
    ):
        raise ValueError("Original isolated measured spectrum pair rows required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "pair_rows_sha256": digest(CACHE / "training_rows.parquet"),
            "energy_audit_sha256": digest(previous / "report.json"),
            "split_source_sha256": digest(
                "casmi_ml/forward_diagnostic_training_audit.py"
            ),
            "feature_source_sha256": digest(
                "casmi_ml/full_gap_candidate_compatibility.py"
            ),
            "rule": "Original60000 molecule/120000 same-ion pair slots, retain only recorded singleton finite nonnegative CE (unique values, no merged energy averaging). Same372 molecule-hash~80/20 split; all rows per molecule on same side. Each molecule equal total weight via averaging sufficient statistics per eligible measured row slots. Baseline2048fp + fit-only adduct/instrument onehot + CE/100; interaction arm adds2048 fp*(CE/100), same fixed.01 ridge/unpenalized intercept/clamp0..1. No CE-bin, regularizer or architecture selection. Targets original per-row60D diagnostic strengths, no pair averaging; duplicate slots retain original weighting. Critic not used or refit. Only training-internal per-row heldout MSE; no development predictions, ranking or activation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    train = pd.read_parquet(TRAIN)
    data = train_data()
    pairs = pd.read_parquet(CACHE / "training_rows.parquet")
    if pairs.inchikey14.tolist() != data["keys"]:
        raise ValueError("Original60000 fingerprint/observed row order required")
    positions = pairs[["representative_row", "paired_row"]].to_numpy().reshape(-1)
    owner = np.repeat(np.arange(60000), 2)
    rows = train.iloc[positions].reset_index(drop=True)
    if rows.inchikey14.tolist() != np.repeat(pairs.inchikey14.to_numpy(), 2).tolist():
        raise ValueError("Measured spectrum rows differ from molecule identities")
    signatures = [energy_signature(value) for value in rows.collision_energy_ev]
    eligible = np.array([s is not None and len(s) == 1 for s in signatures])
    energies = np.array([s[0] for s in signatures if s is not None and len(s) == 1])
    rows = rows.loc[eligible].reset_index(drop=True)
    owner = owner[eligible]
    holdout_molecule = heldout_mask(data["keys"])
    holdout = holdout_molecule[owner]
    fit = ~holdout
    # A molecule with one eligible slot must have equal fit weight to a molecule
    # with two slots. Duplicate each singleton, preserving the original pair-slot
    # weighting for two-slot molecules and permitting exact unweighted ridge.
    counts = np.bincount(owner, minlength=60000)
    repeat_ids = np.repeat(np.arange(len(owner)), 2 // counts[owner])
    owner = owner[repeat_ids]
    energies = energies[repeat_ids]
    rows = rows.iloc[repeat_ids].reset_index(drop=True)
    holdout, fit = holdout[repeat_ids], fit[repeat_ids]
    contexts, vocab = [], {}
    for field in ("adduct", "instrument_type"):
        values = np.array([category(x) for x in rows[field]])
        classes = sorted(set(values[fit]))
        mapping = {name: i for i, name in enumerate(classes)}
        matrix = np.zeros((len(rows), len(classes) + 1), np.float32)
        matrix[np.arange(len(rows)), [mapping.get(x, len(classes)) for x in values]] = 1
        contexts.append(matrix)
        vocab[field] = classes
    contexts = np.concatenate(contexts, axis=1)
    targets = []
    for start in range(0, len(rows), 1024):
        if c.stopped():
            return None
        targets.extend(
            evidence_features(row, True)
            for row in rows.iloc[start : start + 1024].to_dict("records")
        )
    targets = np.stack(targets)
    arms = {}
    for interaction in (False, True):
        if c.stopped():
            return None
        name = "ce_interaction" if interaction else "additive_ce_context"

        def block(ids, interaction=interaction):
            return feature_block(
                data["fps"][owner[ids]], energies[ids], contexts[ids], interaction
            )

        predicted, model = fit_streaming_ridge(block, targets, fit)
        np.savez_compressed(output / ("private_" + name + ".npz"), **model)
        arms[name] = {
            "feature_dimension": len(model["mean_x"]),
            "heldout_mse": float(
                np.square(predicted[holdout] - targets[holdout]).mean()
            ),
            "channel_heldout_mse": np.square(predicted[holdout] - targets[holdout])
            .mean(0)
            .tolist(),
            "fit_mse": float(np.square(predicted[fit] - targets[fit]).mean()),
        }
        del predicted, model
        torch.cuda.empty_cache()
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "eligible_singleton_ce_molecules": int((counts > 0).sum()),
        "fit_molecules": len(set(owner[fit])),
        "internal_holdout_molecules": len(set(owner[holdout])),
        "equal_molecule_weighted_row_slots": len(owner),
        "heldout_target_used_for_fit": False,
        "train_development_overlap": 0,
        "arms": arms,
        "additive_minus_interaction_heldout_mse": arms["additive_ce_context"][
            "heldout_mse"
        ]
        - arms["ce_interaction"]["heldout_mse"],
        "seconds": time.monotonic() - started,
        "scope": "Training-internal singleton CE measured signature prediction only; model dimensions differ, all hyperparameters fixed. Heldout molecular targets and categories excluded from fitting. Excludes merged/missing energy spectra; not full CASMI coverage, ranking accuracy, independent external acceptance or a deployed spectrum simulator.",
    }
    write_json(output / "report.json", report)
    write_json(output / "private_context_vocabulary.json", vocab)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
