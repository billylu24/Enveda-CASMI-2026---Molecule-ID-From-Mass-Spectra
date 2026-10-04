"""Training-only fixed60D connected-ion refinement capacity probe."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml.data import write_json
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.full_gap_candidate_training import run as prepare
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze

AUDIT = Path("artifacts/research_loop/rounds/0339_connected_fragment_training_audit")
ION_CHANNELS = np.arange(24, 42, 3)


def refine_interactions(interactions, connected):
    interactions = np.asarray(interactions)
    connected = np.asarray(connected)
    if interactions.shape[-1] != 60 or connected.shape != (
        *interactions.shape[:-1],
        12,
    ):
        raise ValueError(
            "Original60 interactions and aligned12 connected support required"
        )
    if not np.isin(connected, [0, 1]).all():
        raise ValueError("Connected support must be binary")
    result = interactions.copy()
    result[..., ION_CHANNELS] *= connected[..., :6]
    return result


def run(output):
    if not Controller().local_only():
        raise ValueError("Connected capacity audit is local-only")
    output = Path(output)
    report = json.loads((AUDIT / "report.json").read_text())
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    if (
        protocol["source_sha256"] != digest("casmi_ml/connected_fragment_audit.py")
        or protocol["training_sha256"] != digest(TRAIN)
        or protocol["negative_table_sha256"] != digest(HARD)
        or report["private_support_sha256"]
        != digest(AUDIT / "private_connected_support.npy")
        or report["train_development_overlap"] != 0
    ):
        raise ValueError("Isolated connected audit provenance changed")
    connected = np.load(AUDIT / "private_connected_support.npy")
    selected = np.concatenate([np.arange(60000)[:, None], np.load(HARD)], axis=1)
    freeze(
        output / "connected_protocol.json",
        {
            "source_sha256": digest(__file__),
            "audit_report_sha256": digest(AUDIT / "report.json"),
            "preparation_source_sha256": digest(
                "casmi_ml/full_gap_candidate_training.py"
            ),
            "rule": "Fixed original60D model: replace only6 extended observed diagnostic-ion interactions by original broad support AND single-bond-connected heavy composition. Keep all loss channels unchanged. Same dimensions, L2.01,total2.0,original critic/mean/sqrt/pinned15/admission margin. Training capacity probe returns/retains original valid60D model; refined weights private, no deployment/dev evaluation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    observed = {}

    def capture(cosines, interactions, scale, mask=False, checkpoint=None):
        original, history = fit_bonus(cosines, interactions, scale, mask, checkpoint)
        refined = refine_interactions(interactions, connected[selected])
        weights, updated = fit_bonus(cosines, refined, scale, mask, checkpoint)
        np.save(output / "private_refined_weights.npy", weights)
        observed.update(
            original=history,
            connected_refinement=updated,
            objective_gain=history["final_objective"] - updated["final_objective"],
            original_unregularized_ce=history["final_objective"]
            - float(0.005 * (original @ original)),
            refined_unregularized_ce=updated["final_objective"]
            - float(0.005 * (weights @ weights)),
            refined_ion_weight_sum=float(weights[ION_CHANNELS].sum()),
            active_refined_ion_channels=int((weights[ION_CHANNELS] > 1e-6).sum()),
            modified_interaction_elements=int((interactions != refined).sum()),
        )
        return original, history

    with patch("casmi_ml.full_gap_candidate_training.fit_bonus", new=capture):
        original = prepare(output)
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "train_development_overlap": 0,
        "capacity_probe": observed,
        "original_preparation": original,
        "scope": "Training-only matched60D capacity, no score model deployment or development qualification. Original valid model retained; changing weights is not an accuracy gain.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
