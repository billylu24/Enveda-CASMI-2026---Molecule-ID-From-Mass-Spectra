"""Local final-system evaluation after exact full181 implementation control."""

import fcntl
import time

from casmi_ml.research_loop import Controller


def wait_for_report(c, rid):
    while True:
        if c.stopped():
            return False
        r = next((r for r in c.read()["rounds"] if r["id"] == rid), {})
        if r.get("status") == "failed":
            raise RuntimeError("Required local control failed: " + rid)
        if r.get("report"):
            from pathlib import Path

            if Path(r["report"]).exists():
                return True
        time.sleep(10)


def main():
    c = Controller()
    with (c.root / "diagnostic_system_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not wait_for_report(c, "0190_full181_inference_control"):
            return
        for model, proposals, rid in [
            (
                "0186_diagnostic_residual_coarse",
                "0188_diagnostic_coarse_proposals",
                "0191_diagnostic_coarse_system",
            ),
            (
                "0187_diagnostic_residual_exact",
                "0189_diagnostic_exact_proposals",
                "0192_diagnostic_exact_system",
            ),
        ]:
            if not wait_for_report(c, proposals):
                return
            out = c.root / "rounds" / rid
            c.register_round(
                rid,
                "diagnostic_residual_system",
                [
                    ".venv/bin/python",
                    "-u",
                    "-m",
                    "casmi_ml.diagnostic_system_evaluation",
                    "--output",
                    str(out),
                    "--encoder",
                    str(c.root / "rounds" / model / "model.pt"),
                    "--proposals",
                    str(
                        c.root / "rounds" / proposals / "corrected_only_proposals.json"
                    ),
                ],
                out / "report.json",
            )
            c.run_auxiliary(rid)
            result = c.evaluate_round(rid)
            c.complete_round(rid)
            print("Local full system gate", rid, result["winner"], flush=True)


if __name__ == "__main__":
    main()
