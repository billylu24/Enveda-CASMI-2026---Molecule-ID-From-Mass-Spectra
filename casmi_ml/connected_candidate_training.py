"""Matched fixed60D connected diagnostic-ion training using frozen preparation."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml.connected_fragment_capacity_audit import AUDIT, refine_interactions
from casmi_ml.data import write_json
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.full_gap_candidate_training import run as prepare
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze


def bind_model(model):
    model.update(
        compatibility_source_sha256=digest(
            "casmi_ml/connected_candidate_compatibility.py"
        ),
        original_compatibility_source_sha256=digest(
            "casmi_ml/full_gap_candidate_compatibility.py"
        ),
        connected_source_sha256=digest("casmi_ml/connected_fragment_audit.py"),
        refinement_source_sha256=digest(
            "casmi_ml/connected_fragment_capacity_audit.py"
        ),
        training_wrapper_source_sha256=digest(__file__),
    )
    return model


def run(output, mask_chemical_channels=False):
    if not Controller().local_only():
        raise ValueError("Connected training is local-only")
    output = Path(output)
    report = json.loads((AUDIT / "report.json").read_text())
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    if (
        protocol["source_sha256"] != digest("casmi_ml/connected_fragment_audit.py")
        or protocol["training_sha256"] != digest(TRAIN)
        or protocol["negative_table_sha256"] != digest(HARD)
        or digest(AUDIT / "private_connected_support.npy")
        != report["private_support_sha256"]
        or report["train_development_overlap"] != 0
    ):
        raise ValueError("Connected isolated training cache changed")
    connected = np.load(AUDIT / "private_connected_support.npy")
    selected = np.concatenate([np.arange(60000)[:, None], np.load(HARD)], axis=1)
    freeze(
        output / "connected_protocol.json",
        {
            "source_sha256": digest(__file__),
            "preparation_source_sha256": digest(
                "casmi_ml/full_gap_candidate_training.py"
            ),
            "audit_report_sha256": digest(AUDIT / "report.json"),
            "score_source_sha256": digest(
                "casmi_ml/connected_candidate_compatibility.py"
            ),
            "mask_chemical_channels": mask_chemical_channels,
            "rule": "Only6 diagnostic ion support channels refined by connected heavy composition; all loss channels/original60D/critic/mean/sqrt/pinned15/2.0budget/L2.01/admission.05 unchanged. Fixed matched chemical/mask training, no dev-selected parameters.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )

    def fit(cosines, interactions, scale, mask=False, checkpoint=None):
        return fit_bonus(
            cosines,
            refine_interactions(interactions, connected[selected]),
            scale,
            mask,
            checkpoint,
        )

    with patch("casmi_ml.full_gap_candidate_training.fit_bonus", new=fit):
        report = prepare(output, mask_chemical_channels)
    model = bind_model(json.loads((output / "model.json").read_text()))
    write_json(output / "model.json", model)
    report.update(
        model_sha256=digest(output / "model.json"),
        scope="Matched60D connected-ion fit only; zero and exact masked complete physical systems required before scientific evaluation.",
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
