"""Matched peak-loss masking after completed chemical identity attribution."""

import fcntl
import json

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.data import write_json
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.diagnostic_next_cycle import complete_system
from casmi_ml.research_loop import Controller


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Peak-loss attribution is local-only")
    with (c.root / "diagnostic_peak_loss_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        masked = wait_for_complete_system(c, "0235_diagnostic_context_mask_system")
        if masked is None:
            return
        precise = wait_for_complete_system(c, "0230_diagnostic_context_exact_system")
        if precise is None:
            return
        rid = "0241_diagnostic_context_precursor_channels"
        out = c.root / "rounds" / rid
        c.register_round(
            rid,
            "representation_diagnostic",
            [
                ".venv-gpu/bin/python",
                "-u",
                "-m",
                "casmi_ml.diagnostic_residual_training",
                "--output",
                str(out),
                "--contextual",
                "--precise",
                "--mask-peak-losses",
            ],
            out / "report.json",
        )
        with (c.root / "gpu.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            c.run_auxiliary(rid)
        c.evaluate_round(rid)
        c.complete_round(rid)
        execute(
            c,
            "0242_diagnostic_context_precursor_proposals",
            "casmi_ml.chembl_prior_proposals",
            ["--all-queries", "--encoder", str(out / "model.pt")],
            "candidate_diagnostic",
        )
        selected = complete_system(
            c,
            rid,
            "0242_diagnostic_context_precursor_proposals",
            "0243_diagnostic_context_precursor_system",
        )
        if selected is None:
            return
        reports = {
            name: {
                mode: json.loads((source / "report.json").read_text())[mode][
                    "diagnostic_residual"
                ]
                for mode in ("unknown", "known")
            }
            for name, source in [
                ("all_channels", precise),
                ("all_identity_masked", masked),
                ("peak_loss_masked", selected),
            ]
        }
        write_json(
            "results/research_loop/diagnostic_peak_loss_attribution.json",
            {
                "local_only": True,
                "independent_acceptance": False,
                "arms": reports,
                "source_rounds": {
                    "all_channels": precise.name,
                    "all_identity_masked": masked.name,
                    "peak_loss_masked": selected.name,
                },
                "scope": "Matched24 chemical dimensions, fixed context/capacity/representative spectra/steps/strength gate; only peak_loss identities zeroed. Repeated development attribution, no new independent holdout or cold inference claim.",
            },
        )


if __name__ == "__main__":
    main()
