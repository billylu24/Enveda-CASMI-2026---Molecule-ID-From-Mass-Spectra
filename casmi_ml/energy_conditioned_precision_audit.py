"""Paired molecule-bootstrap precision audit of fixed train-heldout conditional models."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml import energy_conditioned_candidate_audit as base
from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.data import write_json
from casmi_ml.dino_energy_pairs import energy_signature
from casmi_ml.direct_experiment import train_data
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import TRAIN, freeze

PREVIOUS = Path(
    "artifacts/research_loop/rounds/0384_energy_conditioned_candidate_audit"
)


def bootstrap_summary(numerators, denominators, seed=20261003, repetitions=10000):
    numerators, denominators = (
        np.asarray(x, np.float64) for x in (numerators, denominators)
    )
    if (
        numerators.ndim != 1
        or numerators.shape != denominators.shape
        or not len(numerators)
        or not np.isfinite(numerators).all()
        or not np.isfinite(denominators).all()
        or (denominators < 0).any()
        or denominators.sum() <= 0
    ):
        raise ValueError(
            "Aligned finite molecular contributions with positive denominator required"
        )
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repetitions, 100):
        sampled = rng.integers(
            0, len(numerators), (min(100, repetitions - start), len(numerators))
        )
        denominator = denominators[sampled].sum(1)
        if (denominator <= 0).any():
            raise ValueError("Empty bootstrap denominator")
        draws.extend((numerators[sampled].sum(1) / denominator).tolist())
    return {
        "molecules": len(numerators),
        "paired_difference": float(numerators.sum() / denominators.sum()),
        "paired_molecule_bootstrap95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "molecules_with_positive_contribution": int((numerators > 0).sum()),
        "molecules_with_negative_contribution": int((numerators < 0).sum()),
        "zero_contribution_molecules": int((numerators == 0).sum()),
    }


def run(output):
    output = Path(output)
    prior = json.loads((PREVIOUS / "protocol.json").read_text())
    expected = json.loads((PREVIOUS / "report.json").read_text())
    if prior["source_sha256"] != digest(base.__file__):
        raise ValueError("Exact current-source completed candidate audit required")
    observations = []
    predictions = {False: [], True: []}
    contrasts = []
    original_feature, original_predict, original_contrast = (
        base.evidence_features,
        base.candidate_prediction,
        base.contrast_summary,
    )

    def evidence(row, precise=True):
        value = original_feature(row, precise)
        observations.append(value)
        return value

    def predict(fps, energies, contexts, model, interaction):
        value = original_predict(fps, energies, contexts, model, interaction)
        predictions[interaction].extend(value.reshape(-1, 16, 60)[:, 0])
        return value

    def contrast(positive, negative, compatible):
        contrasts.append(
            (np.asarray(positive), np.asarray(negative), np.asarray(compatible))
        )
        return original_contrast(positive, negative, compatible)

    def bound_freeze(path, value):
        value.update(
            base_candidate_source_sha256=value["source_sha256"],
            source_sha256=digest(__file__),
            previous_candidate_report_sha256=digest(PREVIOUS / "report.json"),
            rule=value["rule"]
            + " Reconstruct exact384 contrasts without refit, then paired bootstrap by whole eligible molecular identity, fixedseed20261003/10000draws. Additive minus interaction MSE and interaction minus additive pair-margin; repeated training-internal diagnostics only, no dev/activation.",
        )
        freeze(path, value)

    with (
        patch.object(base, "evidence_features", new=evidence),
        patch.object(base, "candidate_prediction", new=predict),
        patch.object(base, "contrast_summary", new=contrast),
        patch.object(base, "freeze", new=bound_freeze),
    ):
        report = base.run(output)
    if report is None:
        return None
    if report["arms"] != expected["arms"] or len(contrasts) != 2:
        raise ValueError(
            "Original384 complete candidate contrasts must reconstruct exactly"
        )
    data = train_data()
    pairs = pd.read_parquet(CACHE / "training_rows.parquet")
    positions = pairs[["representative_row", "paired_row"]].to_numpy().reshape(-1)
    energies = (
        pd.read_parquet(TRAIN, columns=["collision_energy_ev"])
        .iloc[positions]
        .collision_energy_ev
    )
    eligible = np.array(
        [
            value is not None and len(value) == 1
            for value in map(energy_signature, energies)
        ]
    )
    owner = np.repeat(np.arange(60000), 2)[eligible]
    counts = np.bincount(owner, minlength=60000)
    owner = owner[np.repeat(np.arange(len(owner)), 2 // counts[owner])]
    owner = owner[heldout_mask(data["keys"])[owner]]
    identities, inverse = np.unique(owner, return_inverse=True)
    observed = np.stack(observations)
    if len(observed) != len(owner) or any(
        len(p) != len(owner) for p in predictions.values()
    ):
        raise ValueError("Original full heldout row ordering required")
    loss = [
        np.square(np.stack(predictions[key]) - observed).mean(1)
        for key in (False, True)
    ]
    numerator = np.bincount(
        inverse, weights=loss[0] - loss[1], minlength=len(identities)
    )
    denominator = np.bincount(inverse, minlength=len(identities))
    precision = {
        "additive_minus_interaction_mse": bootstrap_summary(numerator, denominator)
    }
    pa, na, allowed = contrasts[0]
    pb, nb, other_allowed = contrasts[1]
    if not np.array_equal(allowed, other_allowed):
        raise ValueError("Matched identical heldout mass denominator required")
    delta = ((pb[:, None] - nb) - (pa[:, None] - na)) * allowed
    numerator = np.bincount(inverse, weights=delta.sum(1), minlength=len(identities))
    denominator = np.bincount(
        inverse, weights=allowed.sum(1), minlength=len(identities)
    )
    precision["interaction_minus_additive_same_query_pair_margin"] = bootstrap_summary(
        numerator, denominator
    )
    report.update(
        precision=precision,
        bootstrap_seed=20261003,
        bootstrap_samples=10000,
        scope="Exact original384 fixed-model train-heldout contrast reconstruction with molecule-block bootstrap for MSE and conditional positive-negative pair margin. Counts are whole eligible molecules for bootstrap. Precision conditional on repeatedly inspected training split; no external acceptance, calibrated full ranking, dev score or new model activation.",
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
