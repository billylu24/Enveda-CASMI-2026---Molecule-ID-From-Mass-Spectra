"""Matched fixed120D complementary diagnostic support training, no dev selection."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml import full_gap_candidate_training as base
from casmi_ml.data import write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.forward_candidate_capacity_audit import predicted_interactions
from casmi_ml.forward_diagnostic_training_audit import fit_ridge
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

AUDIT = Path("artifacts/research_loop/rounds/0374_forward_augmented_capacity_audit")


def bind_model(model, forward_path):
    model.update(
        weights=[0.0] * 120 if not any(model["weights"]) else model["weights"],
        dimension=120,
        compatibility_source_sha256=digest(
            "casmi_ml/forward_augmented_compatibility.py"
        ),
        original_compatibility_source_sha256=digest(
            "casmi_ml/full_gap_candidate_compatibility.py"
        ),
        forward_source_sha256=digest("casmi_ml/forward_diagnostic_training_audit.py"),
        training_wrapper_source_sha256=digest(__file__),
        forward_model=str(Path(forward_path).resolve()),
        forward_model_sha256=digest(forward_path),
    )
    return model


def run(output, mask_chemical_channels=False):
    if not Controller().local_only():
        raise ValueError("Complementary diagnostic training is local-only")
    output = Path(output)
    probe = json.loads((AUDIT / "report.json").read_text())
    if probe["capacity_probe"]["same_objective_gain"] <= 0:
        raise ValueError("Completed positive train capacity evidence required")
    freeze(
        output / "forward_training_protocol.json",
        {
            "source_sha256": digest(__file__),
            "capacity_audit_sha256": digest(AUDIT / "report.json"),
            "capacity_source_sha256": digest(
                "casmi_ml/forward_augmented_capacity_audit.py"
            ),
            "score_source_sha256": digest(
                "casmi_ml/forward_augmented_compatibility.py"
            ),
            "mask_chemical_channels": mask_chemical_channels,
            "rule": "Fixed120D original60 binary support plus60 same-query observed strength times fp-predicted diagnostic marginal strength; forward full60K ridgeL2.01/clamp0..1, original pinned15/120K/critic/encoder/mean/sqrt/unrestricted denominator. Same total2/L2.01/200iter and margin.05, no absent-peak penalty. Matched all-chemistry-zero bonus arm fits the same forward signatures and changes only final chemical mask. No dev-selected weights or architecture size.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    evidence = []
    original_evidence = base.evidence_features
    data = train_data()
    selected = np.concatenate([np.arange(60000)[:, None], np.load(HARD)], axis=1)
    forward_path = output / "private_forward_signature_model.npz"

    def record(row, precise=True):
        value = original_evidence(row, precise)
        evidence.append(value)
        return value

    def fit(cosines, interactions, scale, mask=False, checkpoint=None):
        if len(evidence) != 120000:
            raise ValueError("Original120K observed training pair rows required")
        chemical = np.stack(evidence).reshape(60000, 2, 60).mean(1)
        predicted, forward = fit_ridge(data["fps"], chemical, np.ones(60000, bool))
        np.savez_compressed(forward_path, **forward)
        augmented = np.concatenate(
            [interactions, predicted_interactions(chemical, predicted, selected)],
            axis=-1,
        )
        return fit_bonus(cosines, augmented, scale, mask, checkpoint)

    with (
        patch.object(base, "evidence_features", new=record),
        patch.object(base, "fit_bonus", new=fit),
    ):
        report = base.run(output, mask_chemical_channels)
    model = bind_model(json.loads((output / "model.json").read_text()), forward_path)
    if len(model["weights"]) != 120:
        raise ValueError("Matched120D weights required")
    write_json(output / "model.json", model)
    report.update(
        model_sha256=digest(output / "model.json"),
        dimension=120,
        scope="Fixed complementary120D training only. Full source-bound exact zero/mask controls and both3340 physical systems before scientific evaluation or activation.",
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-chemical-channels", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
