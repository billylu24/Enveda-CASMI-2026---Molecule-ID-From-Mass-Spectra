"""Training-internal molecular holdout of bonus calibration, frozen critic unchanged."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from casmi_ml import full_gap_candidate_training as base
from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.data import category, write_json
from casmi_ml.dino_energy_pairs import energy_signature
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.energy_conditioned_capacity_audit import conditional_interactions
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives

FORWARD = Path(
    "artifacts/research_loop/rounds/0383_energy_conditioned_diagnostic_audit"
)


def candidate_losses(scores, scale, allowed=None):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 16 or not np.isfinite(scores).all():
        raise ValueError("Finite positive plus original15 scores required")
    logits = scores * scale
    if allowed is not None:
        allowed = np.asarray(allowed, bool)
        if allowed.shape != scores.shape or not allowed[:, 0].all():
            raise ValueError(
                "Positive retained and aligned eligible negatives required"
            )
        logits = np.where(allowed, logits, -np.inf)
    return logsumexp(logits, axis=1) - logits[:, 0]


def molecular_bootstrap_gain(baseline, candidate, seed=20261004, draws=10000):
    delta = np.asarray(baseline, np.float64) - np.asarray(candidate, np.float64)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Paired finite per-molecule losses required")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, np.float64)
    for start in range(0, draws, 64):
        n = min(64, draws - start)
        ids = rng.integers(0, len(delta), size=(n, len(delta)))
        means[start : start + n] = delta[ids].mean(1)
    return {
        "molecules": len(delta),
        "mean_loss_reduction": float(delta.mean()),
        "paired_bootstrap95_loss_reduction": np.quantile(
            means, [0.025, 0.975]
        ).tolist(),
        "seed": seed,
        "draws": draws,
    }


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only frozen-critic GPU diagnostic required")
    output = Path(output)
    prior = json.loads((FORWARD / "protocol.json").read_text())
    evidence = json.loads((FORWARD / "report.json").read_text())
    if (
        prior["source_sha256"]
        != digest("casmi_ml/energy_conditioned_diagnostic_audit.py")
        or prior["training_sha256"] != digest(TRAIN)
        or prior["split_source_sha256"]
        != digest("casmi_ml/forward_diagnostic_training_audit.py")
        or prior["pair_rows_sha256"] != digest(CACHE / "training_rows.parquet")
        or evidence["heldout_target_used_for_fit"]
    ):
        raise ValueError(
            "Original383 fit-only conditional forward model and split required"
        )
    freeze(
        output / "bonus_generalization_protocol.json",
        {
            "source_sha256": digest(__file__),
            "base_training_sha256": digest(base.__file__),
            "forward_protocol_sha256": digest(FORWARD / "protocol.json"),
            "forward_model_sha256": digest(FORWARD / "private_ce_interaction.npz"),
            "vocabulary_sha256": digest(FORWARD / "private_context_vocabulary.json"),
            "conditional_interactions_sha256": digest(
                "casmi_ml/energy_conditioned_capacity_audit.py"
            ),
            "optimizer_sha256": digest("casmi_ml/full_gap_candidate_bonus.py"),
            "split_sha256": digest("casmi_ml/forward_diagnostic_training_audit.py"),
            "training_sha256": digest(TRAIN),
            "hard_table_sha256": digest(HARD),
            "rule": "Original60K/120K paired measured spectra and pinned15; same372/383 molecule-hash split. Freeze already-trained original encoder/critic (trained on all60K; this is NOT independent end-to-end generalization). Bonus fit uses ONLY fit-query molecules for binary60 and binary60+conditional60 separately; .01L2/sum2/SLSQP200/original critic scale/denominator unchanged. Conditional signature weights/context vocabulary reused from383 fit-only molecular split; no heldout spectrum target in forward or bonus fitting. Missing/merged CE extras0, original binary60 retained. Evaluate all12006 heldout query molecules under original15 denominator and separate both-heldout mass-compatible negatives. No development metrics, hyperparameter selection or activation. Retain original valid all-training60D output model, private diagnostic fits only.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    data = train_data()
    pairs = pd.read_parquet(CACHE / "training_rows.parquet")
    if pairs.inchikey14.tolist() != data["keys"]:
        raise ValueError("Original measured molecule order differs")
    holdout = heldout_mask(data["keys"])
    fit_queries = ~holdout
    if int(holdout.sum()) != 12006 or int(fit_queries.sum()) != 47994:
        raise ValueError("Pinned original molecular split changed")
    selected = np.concatenate([np.arange(60000)[:, None], np.load(HARD)], axis=1)
    train = pd.read_parquet(TRAIN)
    rows = train.iloc[
        pairs[["representative_row", "paired_row"]].to_numpy().reshape(-1)
    ]
    signatures = [energy_signature(x) for x in rows.collision_energy_ev]
    energies = np.array(
        [s[0] if s is not None and len(s) == 1 else np.nan for s in signatures]
    )
    vocab = json.loads((FORWARD / "private_context_vocabulary.json").read_text())
    contexts = []
    for field in ("adduct", "instrument_type"):
        mapping = {name: i for i, name in enumerate(vocab[field])}
        matrix = np.zeros((len(rows), len(mapping) + 1), np.float32)
        matrix[
            np.arange(len(rows)),
            [mapping.get(category(x), len(mapping)) for x in rows[field]],
        ] = 1
        contexts.append(matrix)
    contexts = np.concatenate(contexts, axis=1).reshape(60000, 2, -1)
    with np.load(FORWARD / "private_ce_interaction.npz") as archive:
        model = {name: archive[name] for name in ("weights", "mean_x", "mean_y")}
    with torch.inference_mode():
        fps = torch.as_tensor(data["fps"], dtype=torch.float64, device="cuda")
        weights = torch.as_tensor(model["weights"], dtype=torch.float64, device="cuda")
        projections = (
            (fps @ weights[:2048]).cpu().numpy(),
            (fps @ weights[-2048:]).cpu().numpy(),
        )
    compatible = admissible_negatives(data["masses"], selected[:, 1:])
    restricted = np.concatenate(
        [
            np.ones((int(holdout.sum()), 1), bool),
            compatible[holdout] & holdout[selected[holdout, 1:]],
        ],
        axis=1,
    )
    observed = []
    capture = {}
    original_evidence = base.evidence_features

    def record(row, precise=True):
        value = original_evidence(row, precise)
        observed.append(value)
        return value

    def fitter(cosines, interactions, scale, mask=False, checkpoint=None):
        original, history = fit_bonus(cosines, interactions, scale, mask, checkpoint)
        if len(observed) != 120000:
            raise ValueError("All original measured paired observations required")
        extra = conditional_interactions(
            np.stack(observed).reshape(60000, 2, 60),
            energies.reshape(60000, 2),
            contexts,
            projections,
            model,
            selected,
        )
        augmented = np.concatenate([interactions, extra], axis=-1)
        fits = {}
        for name, features in [
            ("binary60", interactions),
            ("conditional120", augmented),
        ]:
            w, h = fit_bonus(
                cosines[fit_queries], features[fit_queries], scale, mask, checkpoint
            )
            np.save(output / ("private_" + name + "_weights.npy"), w)
            scores = cosines[holdout] + features[holdout] @ w
            loss = candidate_losses(scores, scale)
            limited = candidate_losses(scores, scale, restricted)
            capture[name] = {
                "fit_objective": h,
                "heldout_original15_cross_entropy": float(loss.mean()),
                "heldout_both_mass_compatible_cross_entropy": float(limited.mean()),
                "weight_sum": float(w.sum()),
                "active_weights": int((w > 1e-6).sum()),
            }
            fits[name] = (loss, limited)
        zero = candidate_losses(cosines[holdout], scale)
        zero_limited = candidate_losses(cosines[holdout], scale, restricted)
        capture["frozen_critic"] = {
            "heldout_original15_cross_entropy": float(zero.mean()),
            "heldout_both_mass_compatible_cross_entropy": float(zero_limited.mean()),
        }
        capture["binary60_minus_conditional120"] = {
            "original15": molecular_bootstrap_gain(
                fits["binary60"][0], fits["conditional120"][0]
            ),
            "both_heldout_mass_compatible": molecular_bootstrap_gain(
                fits["binary60"][1], fits["conditional120"][1]
            ),
        }
        capture["critic_minus_binary60"] = molecular_bootstrap_gain(
            zero, fits["binary60"][0]
        )
        capture["critic_minus_conditional120"] = molecular_bootstrap_gain(
            zero, fits["conditional120"][0]
        )
        return original, history

    with (
        patch.object(base, "evidence_features", new=record),
        patch.object(base, "fit_bonus", new=fitter),
    ):
        prepared = base.run(output)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "fit_query_molecules": int(fit_queries.sum()),
        "heldout_query_molecules": int(holdout.sum()),
        "heldout_query_targets_used_for_bonus_fit": False,
        "heldout_targets_used_for_forward_fit": False,
        "frozen_critic_pretrained_on_all60000": True,
        "restricted_pair_slots": int(restricted[:, 1:].sum()),
        "restricted_queries_with_negatives": int(restricted[:, 1:].any(axis=1).sum()),
        "calibration": capture,
        "original_valid_model_sha256": digest(output / "model.json"),
        "original_training_report": prepared,
        "scope": "Training-internal bonus-calibration molecular holdout, shared frozen critic trained on all60K. Private bonus fits not activated, no development or external accuracy acceptance. No fit/heldout objective comparison; loss reductions are paired on SAME heldout scores/denominator.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
