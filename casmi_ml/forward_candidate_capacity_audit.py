"""Training-only fixed60D capacity probe of predicted diagnostic candidate support."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml import full_gap_candidate_training as base
from casmi_ml.data import write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.forward_diagnostic_training_audit import fit_ridge
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

AUDIT = Path("artifacts/research_loop/rounds/0372_forward_diagnostic_training_audit")


def predicted_interactions(chemical, predicted, selected):
    chemical = np.asarray(chemical, np.float64)
    predicted = np.asarray(predicted, np.float64)
    selected = np.asarray(selected)
    if (
        chemical.shape != predicted.shape
        or chemical.ndim != 2
        or chemical.shape[1] != 60
    ):
        raise ValueError(
            "Aligned original measured60D and predicted60D signatures required"
        )
    if selected.shape != (len(chemical), 16) or not np.issubdtype(
        selected.dtype, np.integer
    ):
        raise ValueError("Original positive+15 pinned candidate indices required")
    if (selected < 0).any() or (selected >= len(chemical)).any():
        raise ValueError("Invalid selected training molecular identity")
    if (
        not np.isfinite(chemical).all()
        or not np.isfinite(predicted).all()
        or (chemical < 0).any()
        or (chemical > 1).any()
        or (predicted < 0).any()
        or (predicted > 1).any()
    ):
        raise ValueError(
            "Bounded finite measured/predicted diagnostic strengths required"
        )
    return chemical[:, None, :] * predicted[selected]


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Forward capacity probe is local-only")
    audit = json.loads((AUDIT / "report.json").read_text())
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    if (
        protocol["source_sha256"]
        != digest("casmi_ml/forward_diagnostic_training_audit.py")
        or audit["internal_holdout_target_used_for_fit"]
        or audit["train_development_overlap"]
    ):
        raise ValueError(
            "Source-bound isolated training-internal forward audit required"
        )
    if (
        audit["arms"]["fingerprint_ridge"]["heldout_mse"]
        >= audit["arms"]["broad_motif_ridge"]["heldout_mse"]
    ):
        raise ValueError("No matched internal holdout prediction benefit")
    output = Path(output)
    freeze(
        output / "forward_capacity_protocol.json",
        {
            "source_sha256": digest(__file__),
            "base_training_source_sha256": digest(base.__file__),
            "forward_prediction_source_sha256": digest(
                "casmi_ml/forward_diagnostic_training_audit.py"
            ),
            "heldout_audit_sha256": digest(AUDIT / "report.json"),
            "pinned_table_sha256": digest(HARD),
            "rule": "Original60K/120K/60D/pinned15/mean/sqrt/frozen encoder+critic/sum2/L2.01/200iter. Only replace binary candidate SMARTS60 support with original2048fingerprint->observed mean diagnostic60D fixed ridge L2.01 predictions clamped0..1. Ridge fitted on full original60K, without dev targets;372 internal heldout signal is prerequisite. Same measured query chemistry, positive-only features, no absent-peak penalty. Same original unrestricted denominator, no mass-mask combination. Compare identical convex objectives; save private predicted-support model and private new bonus weights but return original valid model. No development evaluation or activation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    evidence = []
    captured = {}
    original_evidence = base.evidence_features
    data = train_data()
    hard = np.load(HARD)
    selected = np.concatenate([np.arange(60000)[:, None], hard], axis=1)

    def record(row, precise=True):
        value = original_evidence(row, precise)
        evidence.append(value)
        return value

    def fit(cosines, interactions, scale, mask=False, checkpoint=None):
        old, history = fit_bonus(cosines, interactions, scale, mask, checkpoint)
        if len(evidence) != 120000:
            raise ValueError("Exact original120K measured evidence rows required")
        chemical = np.stack(evidence).reshape(60000, 2, 60).mean(1)
        predicted, forward = fit_ridge(data["fps"], chemical, np.ones(60000, bool))
        np.savez_compressed(output / "private_forward_signature_model.npz", **forward)
        new_interactions = predicted_interactions(chemical, predicted, selected)
        weights, updated = fit_bonus(cosines, new_interactions, scale, mask, checkpoint)
        np.save(output / "private_forward_bonus_weights.npy", weights)
        captured.update(
            original=history,
            forward_candidate=updated,
            same_objective_gain=history["final_objective"] - updated["final_objective"],
            active_forward_channels=int((weights > 1e-6).sum()),
            private_forward_sha256=digest(
                output / "private_forward_signature_model.npz"
            ),
            private_bonus_sha256=digest(output / "private_forward_bonus_weights.npy"),
        )
        return old, history

    with (
        patch.object(base, "evidence_features", new=record),
        patch.object(base, "fit_bonus", new=fit),
    ):
        prepared = base.run(output)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "capacity_probe": captured,
        "original_preparation": prepared,
        "scope": "Fixed original60D capacity test, forward marginal diagnostic strength is not charge/energy-specific spectrum simulation. Full-training objective benefit can overfit and is not development accuracy. Original valid model retained; new private weights never deployed. Requires matched complete controls and full systems before scientific evaluation.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
