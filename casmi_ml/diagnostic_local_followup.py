"""Sequential local-only residual evaluation after independent complete control audit."""

import fcntl
import json
from pathlib import Path

from casmi_ml.diagnostic_system_followup import wait_for_report
from casmi_ml.research_loop import Controller


def execute(c, rid, module, arguments, direction):
    out = c.root / "rounds" / rid
    c.register_round(
        rid,
        direction,
        [".venv/bin/python", "-u", "-m", module, "--output", str(out), *arguments],
        out / "report.json",
    )
    c.run_auxiliary(rid)
    report = out / "report.json"
    if not report.exists():
        raise RuntimeError("Required local evaluation did not produce report: " + rid)
    result = c.evaluate_round(rid)
    c.complete_round(rid)
    return out, result


def main():
    c = Controller()
    if not c.local_only():
        raise RuntimeError("This driver requires persistent local-only mode")
    with (c.root / "diagnostic_local_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not wait_for_report(c, "0216_full181_all_physical_control_attribution"):
            return
        audit = c.root / "rounds/0216_full181_all_physical_control_attribution"
        if not json.loads((audit / "report.json").read_text()).get("control_passed"):
            raise RuntimeError("Independent complete control attribution failed")
        if c.stopped():
            return
        execute(
            c,
            "0206_full181_audited_exact_control",
            "casmi_ml.diagnostic_system_evaluation",
            [
                "--encoder",
                "artifacts/scale_20260929/runs/wide_enhanced_60000/model.pt",
                "--proposals",
                str(
                    c.root
                    / "rounds/0147_chembl_all_query_proposals/corrected_only_proposals.json"
                ),
                "--control",
                "--control-reference",
                str(audit),
                "--reuse-unknown",
                str(c.root / "rounds/0204_full181_complete_control_audit"),
            ],
            "implementation_control",
        )
        for model, proposals, rid in [
            (
                "0186_diagnostic_residual_coarse",
                "0188_diagnostic_coarse_proposals",
                "0207_diagnostic_coarse_system",
            ),
            (
                "0187_diagnostic_residual_exact",
                "0189_diagnostic_exact_proposals",
                "0208_diagnostic_exact_system",
            ),
        ]:
            if c.stopped():
                return
            if not c.local_only():
                raise RuntimeError("Local-only mode changed during evaluation")
            _, result = execute(
                c,
                rid,
                "casmi_ml.diagnostic_system_evaluation",
                [
                    "--encoder",
                    str(c.root / "rounds" / model / "model.pt"),
                    "--proposals",
                    str(
                        c.root / "rounds" / proposals / "corrected_only_proposals.json"
                    ),
                    "--paired-control",
                    str(c.root / "rounds/0206_full181_audited_exact_control"),
                ],
                "diagnostic_residual_system",
            )
            print("Local full system gate", rid, result.get("winner"), flush=True)
        Path(
            "results/research_loop/diagnostic_residual_local_followup.json"
        ).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "local_only": True,
                    "control": "0206_full181_audited_exact_control",
                    "arms": [
                        "0207_diagnostic_coarse_system",
                        "0208_diagnostic_exact_system",
                    ],
                    "remote_publication": False,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
