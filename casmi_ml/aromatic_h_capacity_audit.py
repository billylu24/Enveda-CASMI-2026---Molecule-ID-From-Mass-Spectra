"""Training-only matched60D capacity: three aromatic diagnostic-ion H refinements."""

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

AUDIT = Path("artifacts/research_loop/rounds/0346_fragment_hydrogen_training_audit")
CHANNELS = np.array([27, 30, 33])


def refine_interactions(interactions, exact_h):
    interactions, exact_h = np.asarray(interactions), np.asarray(exact_h)
    if interactions.shape[-1] != 60 or exact_h.shape != (*interactions.shape[:-1], 6):
        raise ValueError("Original60D and aligned six-ion H support required")
    if not np.isin(exact_h, [0, 1]).all():
        raise ValueError("Binary support required")
    result = interactions.copy()
    result[..., CHANNELS] *= exact_h[..., 1:4]
    return result


def run(output):
    if not Controller().local_only():
        raise ValueError("Aromatic H capacity probe is local-only")
    output = Path(output)
    report = json.loads((AUDIT / "report.json").read_text())
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    if (
        protocol["source_sha256"] != digest("casmi_ml/fragment_hydrogen_audit.py")
        or protocol["training_sha256"] != digest(TRAIN)
        or protocol["negative_table_sha256"] != digest(HARD)
        or report["train_development_overlap"] != 0
    ):
        raise ValueError("Isolated H audit source changed")
    name = "private_exact_h_support.npy"
    if digest(AUDIT / name) != report["private_input_sha256"][name]:
        raise ValueError("Exact H support changed")
    exact = np.load(AUDIT / name)
    selected = np.concatenate([np.arange(60000)[:, None], np.load(HARD)], axis=1)
    freeze(
        output / "aromatic_h_protocol.json",
        {
            "source_sha256": digest(__file__),
            "audit_report_sha256": digest(AUDIT / "report.json"),
            "preparation_source_sha256": digest(
                "casmi_ml/full_gap_candidate_training.py"
            ),
            "rule": "Original60D plus no new dimensions: refine onlybenzyl/tropylium,benzoyl,benzoate existing diagnostic-ion channels by connected heavy composition AND exact original graph H.346 training-only positive/negative ratios motivate this subset; phenyl direction weak and phosphate deprotonation violates original H so preserve their original broad support. All loss channels/critic/mean/sqrt/pinned15/L2.01/2.0budget/margin.05 unchanged. Probe retains original valid60D model, new weights private, no dev fitting/score or deployment.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    observed = {}

    def capture(cosines, interactions, scale, mask=False, checkpoint=None):
        original, history = fit_bonus(cosines, interactions, scale, mask, checkpoint)
        refined = refine_interactions(interactions, exact[selected])
        weights, updated = fit_bonus(cosines, refined, scale, mask, checkpoint)
        np.save(output / "private_refined_weights.npy", weights)
        observed.update(
            original=history,
            aromatic_h_refinement=updated,
            objective_gain=history["final_objective"] - updated["final_objective"],
            original_unregularized_ce=history["final_objective"]
            - float(0.005 * (original @ original)),
            refined_unregularized_ce=updated["final_objective"]
            - float(0.005 * (weights @ weights)),
            refined_ion_weight_sum=float(weights[CHANNELS].sum()),
            active_refined_ion_channels=int((weights[CHANNELS] > 1e-6).sum()),
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
        "scope": "Training-only matched60D aromatic original-H refinement capacity. No scored development or model activation; training objective is not accuracy proof.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
