"""Train-only120D conditioned diagnostic support capacity probe, retain original model."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from casmi_ml import full_gap_candidate_training as base
from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.data import category, write_json
from casmi_ml.dino_energy_pairs import energy_signature
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.energy_conditioned_diagnostic_audit import (
    feature_block,
    fit_streaming_ridge,
)
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze

AUDIT = Path("artifacts/research_loop/rounds/0383_energy_conditioned_diagnostic_audit")
CONTRAST = Path(
    "artifacts/research_loop/rounds/0384_energy_conditioned_candidate_audit"
)


def conditional_interactions(
    observed, energies, contexts, projections, model, selected
):
    observed, energies, contexts = map(np.asarray, (observed, energies, contexts))
    if (
        observed.ndim != 3
        or observed.shape[1:] != (2, 60)
        or energies.shape != observed.shape[:2]
        or contexts.shape[:2] != observed.shape[:2]
    ):
        raise ValueError(
            "Original paired60D observations and per-row contexts required"
        )
    if not np.isfinite(observed).all() or (observed < 0).any() or (observed > 1).any():
        raise ValueError("Bounded observed diagnostic strength required")
    original, energy = projections
    if (
        original.shape != energy.shape
        or original.shape != (len(observed), 60)
        or selected.shape != (len(observed), 16)
    ):
        raise ValueError("Aligned candidate projections and original pinned15 required")
    weights = model["weights"]
    context_weights = weights[2048:-2049]
    energy_weights = weights[-2049]
    bias = model["mean_y"] - model["mean_x"] @ weights
    result = np.zeros((len(observed), 16, 60), np.float64)
    for start in range(0, len(observed), 256):
        stop = min(start + 256, len(observed))
        ids = selected[start:stop]
        for slot in range(2):
            ce = energies[start:stop, slot]
            eligible = np.isfinite(ce) & (ce >= 0)
            ce = np.where(eligible, ce / 100, 0)
            context = (
                contexts[start:stop, slot] @ context_weights
                + ce[:, None] * energy_weights
                + bias
            )
            predicted = np.clip(
                original[ids] + energy[ids] * ce[:, None, None] + context[:, None, :],
                0,
                1,
            )
            result[start:stop] += (
                observed[start:stop, slot, None, :]
                * predicted
                * eligible[:, None, None]
                / 2
            )
    return result


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU capacity probe required")
    output = Path(output)
    audit = json.loads((AUDIT / "report.json").read_text())
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    contrast = json.loads((CONTRAST / "report.json").read_text())
    if (
        protocol["source_sha256"]
        != digest("casmi_ml/energy_conditioned_diagnostic_audit.py")
        or audit["heldout_target_used_for_fit"]
        or audit["additive_minus_interaction_heldout_mse"] <= 0
    ):
        raise ValueError(
            "Fixed train-internal heldout conditional prediction gain required"
        )
    if (
        contrast["arms"]["ce_interaction"]["mean_positive_minus_negative"]
        <= contrast["arms"]["additive_ce_context"]["mean_positive_minus_negative"]
    ):
        raise ValueError("No positive same-query heldout pair direction")
    freeze(
        output / "conditional_capacity_protocol.json",
        {
            "source_sha256": digest(__file__),
            "base_training_source_sha256": digest(base.__file__),
            "prior_source_sha256": protocol["source_sha256"],
            "heldout_prediction_report_sha256": digest(AUDIT / "report.json"),
            "contrast_report_sha256": digest(CONTRAST / "report.json"),
            "training_sha256": digest(TRAIN),
            "rule": "Original60000/120000 same-ion paired rows/pinned15/frozen encoder+critic/unrestricted denominator;120D original60binary structural support plus60 mean per-row observed diagnostic strength times candidate predictions conditioned on SAME query singletonCE/adduct/instrument. Missing/merged CE rows contribute zero extra but original60 untouched; divide by original2slots. Full original training singletonCE ridge .01/equal eligible molecule weight/clamp0..1, original total2/L2.01/200iter for bonus. Factorized conditional linear model for candidate projections, float64 then interactions as original optimizer float64. Compare same convex objective, private new weights only, return original60D valid model. No dev evaluation, multiplier/bin/architecture tuning or automatic activation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    data = train_data()
    selected = np.concatenate([np.arange(60000)[:, None], np.load(HARD)], axis=1)
    pairs = pd.read_parquet(CACHE / "training_rows.parquet")
    if (
        protocol["pair_rows_sha256"] != digest(CACHE / "training_rows.parquet")
        or pairs.inchikey14.tolist() != data["keys"]
    ):
        raise ValueError("Original measured training pair binding differs")
    train = pd.read_parquet(TRAIN)
    rows = train.iloc[
        pairs[["representative_row", "paired_row"]].to_numpy().reshape(-1)
    ].reset_index(drop=True)
    owner = np.repeat(np.arange(60000), 2)
    signatures = [energy_signature(x) for x in rows.collision_energy_ev]
    energies = np.array(
        [x[0] if x is not None and len(x) == 1 else np.nan for x in signatures]
    )
    eligible = np.isfinite(energies)
    contexts = []
    vocab = {}
    for field in ("adduct", "instrument_type"):
        values = np.array([category(x) for x in rows[field]])
        classes = sorted(set(values[eligible]))
        vocab[field] = classes
        mapping = {name: i for i, name in enumerate(classes)}
        matrix = np.zeros((len(rows), len(classes) + 1), np.float32)
        matrix[np.arange(len(rows)), [mapping.get(x, len(classes)) for x in values]] = 1
        contexts.append(matrix)
    contexts = np.concatenate(contexts, axis=1)
    counts = np.bincount(owner[eligible], minlength=60000)
    eligible_ids = np.flatnonzero(eligible)
    fit_ids = np.repeat(eligible_ids, 2 // counts[owner[eligible]])
    observed = []
    capture = {}
    original_evidence = base.evidence_features

    def record(row, precise=True):
        value = original_evidence(row, precise)
        observed.append(value)
        return value

    def fit(cosines, interactions, scale, mask=False, checkpoint=None):
        if len(observed) != 120000:
            raise ValueError("Original120000 measured pair slots required")
        old, history = fit_bonus(cosines, interactions, scale, mask, checkpoint)
        chemical = np.stack(observed)

        def block(ids):
            use = fit_ids[ids]
            return feature_block(
                data["fps"][owner[use]], energies[use], contexts[use], True
            )

        unused, forward = fit_streaming_ridge(
            block, chemical[fit_ids], np.ones(len(fit_ids), bool)
        )
        del unused
        np.savez_compressed(
            output / "private_conditional_signature_model.npz", **forward
        )
        write_json(output / "private_context_vocabulary.json", vocab)
        fps = torch.as_tensor(data["fps"], dtype=torch.float64, device="cuda")
        weights = torch.as_tensor(
            forward["weights"], dtype=torch.float64, device="cuda"
        )
        projections = (
            (fps @ weights[:2048]).cpu().numpy(),
            (fps @ weights[-2048:]).cpu().numpy(),
        )
        new = conditional_interactions(
            chemical.reshape(60000, 2, 60),
            energies.reshape(60000, 2),
            contexts.reshape(60000, 2, -1),
            projections,
            forward,
            selected,
        )
        augmented = np.concatenate([interactions, new], axis=-1)
        weights, updated = fit_bonus(cosines, augmented, scale, mask, checkpoint)
        np.save(output / "private_conditional_bonus_weights.npy", weights)
        capture.update(
            original=history,
            conditioned=updated,
            same_objective_gain=history["final_objective"] - updated["final_objective"],
            active_original_channels=int((weights[:60] > 1e-6).sum()),
            active_conditional_channels=int((weights[60:] > 1e-6).sum()),
            conditional_weight_sum=float(weights[60:].sum()),
            eligible_training_molecules=int((counts > 0).sum()),
        )
        return old, history

    with (
        patch.object(base, "evidence_features", new=record),
        patch.object(base, "fit_bonus", new=fit),
    ):
        prepared = base.run(output)
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "capacity_probe": capture,
        "original_preparation": prepared,
        "scope": "Training-only conditional120D capacity probe; larger model embeds original feasible solution. Internalheldout MSE/pair advantage and training objective do not establish complete ranking gain. Original valid model retained; new weights never activated.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
