"""Training-only convex capacity probe for the binding .05 candidate chemistry cap."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import casmi_ml.bounded_candidate_bonus as solver
from casmi_ml.data import write_json
from casmi_ml.extended_candidate_training import run as prepare
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def run(output):
    if not Controller().local_only():
        raise ValueError("Candidate capacity audit is local-only")
    output = Path(output)
    freeze(
        output / "audit_protocol.json",
        {
            "source_sha256": digest(__file__),
            "preparation_source_sha256": digest(
                "casmi_ml/extended_candidate_training.py"
            ),
            "solver_source_sha256": digest("casmi_ml/bounded_candidate_bonus.py"),
            "rule": "Same frozen original60K/120K measured slots/pinned15 interactions as307, no dev inputs or fit. Fit existing .05 cap, then fixed2.0 cap equal full possible cosine difference[-1,1]. Retain original .01L2/scale/200iter. Capacity diagnostic only; no .05 model/runtime replacement, no development candidate score evaluation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    observed = {}

    def capture(cosines, interactions, scale, mask=False, checkpoint=None):
        # Preserve a valid original .05 model produced by preparation; the larger
        # cap is a private audit artifact and never passed to original runtime.
        bounded, history = solver.fit_bonus(
            cosines, interactions, scale, mask, checkpoint
        )
        with patch.object(solver, "BONUS_BUDGET", 2.0):
            expanded, larger = solver.fit_bonus(
                cosines, interactions, scale, mask, checkpoint
            )
        observed.update(
            bounded=history,
            full_cosine_difference_capacity=larger,
            expanded_weight_sum=float(expanded.sum()),
            active_expanded_channels=int((expanded > 1e-6).sum()),
            largest_expanded_weight=float(expanded.max()),
            scaled_training_objective_gain=history["final_objective"]
            - larger["final_objective"],
        )
        np.save(output / "private_capacity_weights.npy", expanded)
        return bounded, history

    with patch("casmi_ml.extended_candidate_training.fit_bonus", new=capture):
        original = prepare(output)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "capacity_probe": observed,
        "original_preparation": original,
        "scope": "Training-only bound saturation diagnostic. Fixed2 budget is maximum normalized cosine gap, not development-selected multiplier. No new candidate model activation or accuracy claim; any expanded model would require new source/model binding, matched controls and frozen gates.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
