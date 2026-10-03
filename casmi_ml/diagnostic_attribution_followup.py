"""Matched chemical-channel masking attribution after contextual full-system arms."""

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
        raise ValueError("Chemical attribution followup is local-only")
    with (c.root / "diagnostic_attribution_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        precise = wait_for_complete_system(c, "0230_diagnostic_context_exact_system")
        if precise is None:
            return
        coarse = wait_for_complete_system(c, "0229_diagnostic_context_coarse_system")
        if coarse is None:
            return
        if c.stopped():
            return
        rid = "0233_diagnostic_context_mask_control"
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
                "--mask-chemical-channels",
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
            "0234_diagnostic_context_mask_proposals",
            "casmi_ml.chembl_prior_proposals",
            ["--all-queries", "--encoder", str(out / "model.pt")],
            "candidate_diagnostic",
        )
        masked = complete_system(
            c,
            rid,
            "0234_diagnostic_context_mask_proposals",
            "0235_diagnostic_context_mask_system",
        )
        if masked is None:
            return
        reports = {}
        for name, source in [
            ("precise", precise),
            ("coarse", coarse),
            ("precise_channels_masked", masked),
        ]:
            report = json.loads((source / "report.json").read_text())
            reports[name] = {
                mode: report[mode]["diagnostic_residual"]
                for mode in ("unknown", "known")
            }
        write_json(
            "results/research_loop/diagnostic_context_attribution.json",
            {
                "local_only": True,
                "independent_acceptance": False,
                "arms": reports,
                "source_rounds": {
                    "precise": precise.name,
                    "coarse": coarse.name,
                    "precise_channels_masked": masked.name,
                },
                "precise_minus_mask_unknown_mrr25": reports["precise"]["unknown"][
                    "mrr25"
                ]
                - reports["precise_channels_masked"]["unknown"]["mrr25"],
                "scope": "Matched channels masked under same precise strength gate, frozen original latent, model capacity and training budget; repeated development attribution, no fresh holdout",
            },
        )


if __name__ == "__main__":
    main()
