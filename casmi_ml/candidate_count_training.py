"""Fit fixed count interactions using the unchanged full-gap training preparation."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml.candidate_count_capacity_audit import COUNTS, multiplicity_features
from casmi_ml.candidate_count_compatibility import DIM
from casmi_ml.data import write_json
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.full_gap_candidate_training import run as prepare
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def run(output, mask_chemical_channels=False):
    if not Controller().local_only():
        raise ValueError("Count training is local-only")
    output = Path(output)
    counts_report = json.loads((COUNTS / "report.json").read_text())
    if (
        digest(COUNTS / "private_motif_counts.npy")
        != counts_report["private_counts_sha256"]
    ):
        raise ValueError("Training motif-count provenance differs")
    counts = np.load(COUNTS / "private_motif_counts.npy")
    selected = np.concatenate([np.arange(60000)[:, None], np.load(HARD)], axis=1)
    freeze(
        output / "count_protocol.json",
        {
            "source_sha256": digest(__file__),
            "preparation_source_sha256": digest(
                "casmi_ml/full_gap_candidate_training.py"
            ),
            "count_report_sha256": digest(COUNTS / "report.json"),
            "counts_sha256": counts_report["private_counts_sha256"],
            "score_source_sha256": digest("casmi_ml/candidate_count_compatibility.py"),
            "mask_chemical_channels": mask_chemical_channels,
            "rule": "Only60 additional observed motif multiplicity channels versus327; original encoder/critic/pinned15/mean/sqrt/L2.01/total2.0/optimizer and deploymentmargin.05 unchanged. Same complete zero/mask controls and frozen gates, no dev-selected parameters.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    fitted = {}

    def capture(cosines, interactions, scale, mask=False, checkpoint=None):
        original, history = fit_bonus(cosines, interactions, scale, mask, checkpoint)
        extra = multiplicity_features(interactions, counts[selected])
        weights, updated = fit_bonus(
            cosines,
            np.concatenate([interactions, extra], axis=-1),
            scale,
            mask,
            checkpoint,
        )
        fitted.update(weights=weights, history=updated)
        return original, history

    with patch("casmi_ml.full_gap_candidate_training.fit_bonus", new=capture):
        original = prepare(output, mask_chemical_channels)
    model = json.loads((output / "model.json").read_text())
    write_json(output / "original60_model.json", model)
    model.update(
        weights=fitted["weights"].tolist(),
        dimension=DIM,
        objective=fitted["history"],
        compatibility_source_sha256=digest("casmi_ml/candidate_count_compatibility.py"),
        original_compatibility_source_sha256=digest(
            "casmi_ml/full_gap_candidate_compatibility.py"
        ),
        count_source_sha256=digest("casmi_ml/candidate_chemistry_counts_audit.py"),
        multiplicity_source_sha256=digest("casmi_ml/candidate_count_capacity_audit.py"),
        training_wrapper_source_sha256=digest(__file__),
        count_report_sha256=digest(COUNTS / "report.json"),
    )
    write_json(output / "model.json", model)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "train_development_overlap": 0,
        "history": fitted["history"],
        "original_preparation": original,
        "model_sha256": digest(output / "model.json"),
        "scope": "Fixed120D training only, no scientific activation; complete zero and matched chemical/mask systems required.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--mask-chemical-channels", action="store_true")
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
