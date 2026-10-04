"""Training-only probe of candidate chemistry fitted to mass-admissible competitors."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from casmi_ml import full_gap_candidate_training as training
from casmi_ml.data import write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.full_gap_candidate_bonus import fit_bonus as original_fit
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives


def fit_mass_bonus(cosines, interactions, scale, compatible, checkpoint=None):
    cosines, interactions = (
        np.asarray(cosines, np.float64),
        np.asarray(interactions, np.float64),
    )
    compatible = np.asarray(compatible, bool)
    if (
        cosines.shape[1] != 16
        or interactions.shape[:2] != cosines.shape
        or compatible.shape != (len(cosines), 15)
    ):
        raise ValueError(
            "Original positive+15 candidate interactions must align with mass mask"
        )
    if (
        not np.isfinite(cosines).all()
        or not np.isfinite(interactions).all()
        or ((interactions < 0) | (interactions > 1)).any()
        or not 0 < scale <= 50
    ):
        raise ValueError("Finite original critic/chemical inputs required")
    allowed = np.concatenate([np.ones((len(cosines), 1), bool), compatible], axis=1)
    dimension = interactions.shape[-1]

    def objective(weight):
        if checkpoint is not None and not checkpoint():
            raise RuntimeError("Training capacity budget expired")
        logits = (cosines + interactions @ weight) * scale
        logits = np.where(allowed, logits, -np.inf)
        normalizer = logsumexp(logits, axis=1)
        probability = np.exp(logits - normalizer[:, None])
        probability[:, 0] -= 1
        loss = (normalizer - logits[:, 0]).mean() + 0.01 * (weight @ weight) / 2
        gradient = (
            np.einsum("bc,bcd->d", probability, interactions) * scale / len(cosines)
            + 0.01 * weight
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(dimension),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 2.0)] * dimension,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda w: 2.0 - w.sum(),
                "jac": lambda w: -np.ones(dimension),
            }
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Mass-admissible capacity optimizer failed: " + result.message)
    weights = np.maximum(result.x, 0)
    if weights.sum() > 2:
        weights *= 2 / weights.sum()
    initial = objective(np.zeros(dimension))[0]
    final = objective(weights)[0]
    return weights, {
        "initial_objective": float(initial),
        "final_objective": float(final),
        "same_mask_objective_gain": float(initial - final),
        "optimizer_iterations": int(result.nit),
        "weight_sum": float(weights.sum()),
        "compatible_pairs": int(compatible.sum()),
        "queries_with_compatible_negative": int(compatible.any(1).sum()),
        "query_weighting": "All original60000 equal; no-compatible query contributes zero CE, not resampled",
    }


def run(output):
    if not Controller().local_only():
        raise ValueError("Capacity probe is local-only")
    output = Path(output)
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    compatible = admissible_negatives(data["masses"], hard)
    freeze(
        output / "mass_capacity_protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_source_sha256": digest(training.__file__),
            "mass_mask_source_sha256": digest("casmi_ml/strict_mass_negative_mask.py"),
            "pinned_table_sha256": digest(HARD),
            "rule": "Same original60K/paired120K/60chemicalD/pinned15/original frozen critic. Compare fit with original unrestricted denominator versus same logits excluding only training-mass-incompatible pinned negatives using existing35ppm/.006Da criterion. Positive always allowed, all60K retain equal weight, no resampling. Fixedsum2/L2.01/200iter/admissionmargin.05. Probe saves refined private weights but returns original valid unrestricted model; no development evaluation or activation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    captured = {}

    def capture(cosines, interactions, scale, mask=False, checkpoint=None):
        old, history = original_fit(cosines, interactions, scale, mask, checkpoint)
        candidate, updated = fit_mass_bonus(
            cosines, interactions, scale, compatible, checkpoint
        )
        np.save(output / "private_mass_compatible_weights.npy", candidate)
        masked_logits = np.where(
            np.concatenate([np.ones((len(cosines), 1), bool), compatible], axis=1),
            (np.asarray(cosines) + np.asarray(interactions) @ old) * scale,
            -np.inf,
        )
        old_on_mask = float(
            (logsumexp(masked_logits, axis=1) - masked_logits[:, 0]).mean()
            + 0.01 * (old @ old) / 2
        )
        captured.update(
            original=history,
            mass_compatible=updated,
            original_weights_on_same_mask_objective=old_on_mask,
            matched_mask_objective_gain=old_on_mask - updated["final_objective"],
            active_channels=int((candidate > 1e-6).sum()),
        )
        return old, history

    with patch.object(training, "fit_bonus", new=capture):
        prepared = training.run(output)
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "capacity_probe": captured,
        "original_preparation": prepared,
        "scope": "Training-only candidate-loss distribution probe. No new molecular representation, query prediction, development-selected threshold or source activation. If useful, requires separate paired chemical/mask full3340 systems and original controls.",
    }
    write_json(output / "report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
