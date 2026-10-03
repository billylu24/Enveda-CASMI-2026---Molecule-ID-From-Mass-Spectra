"""Wait for both local matched models, then measure all-query candidate coverage."""

import time

from casmi_ml.research_loop import Controller


def main():
    c = Controller()
    arms = [
        ("0186_diagnostic_residual_coarse", "0188_diagnostic_coarse_proposals"),
        ("0187_diagnostic_residual_exact", "0189_diagnostic_exact_proposals"),
    ]
    with (c.root / "diagnostic_followup.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for source, rid in arms:
            model = c.root / "rounds" / source / "model.pt"
            while not (model.parent / "report.json").exists():
                if c.stopped():
                    return
                source_round = next(
                    (r for r in c.read()["rounds"] if r["id"] == source), {}
                )
                if source_round.get("status") == "failed":
                    raise RuntimeError(f"Training failed: {source}")
                time.sleep(10)
            out = c.root / "rounds" / rid
            c.register_round(
                rid,
                "candidate_diagnostic",
                [
                    ".venv/bin/python",
                    "-u",
                    "-m",
                    "casmi_ml.chembl_prior_proposals",
                    "--output",
                    str(out),
                    "--all-queries",
                    "--encoder",
                    str(model),
                ],
                out / "report.json",
            )
            c.run_auxiliary(rid)
            c.evaluate_round(rid)
            c.complete_round(rid)
            print("Completed local all-query coverage", rid, flush=True)


if __name__ == "__main__":
    main()
