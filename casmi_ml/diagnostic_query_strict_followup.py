"""Matched training loss restricted to actual mass-window ambiguities."""

import fcntl
import json

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.data import write_json
from casmi_ml.diagnostic_query_followup import complete_query_system
from casmi_ml.research_loop import Controller


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Strict mass query cycle is local-only")
    with (c.root / "diagnostic_query_strict_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if wait_for_complete_system(c, "0248_diagnostic_query_mask_system") is None:
            return
        original = wait_for_complete_system(c, "0247_diagnostic_query_chemical_system")
        if original is None:
            return
        models = []
        for mask, rid in [
            (False, "0253_query_strict_mass_chemical"),
            (True, "0254_query_strict_mass_mask_control"),
        ]:
            if c.stopped():
                return
            out = c.root / "rounds" / rid
            c.register_round(
                rid,
                "representation_diagnostic",
                [
                    ".venv-gpu/bin/python",
                    "-u",
                    "-m",
                    "casmi_ml.diagnostic_query_training",
                    "--output",
                    str(out),
                    "--strict-mass-negatives",
                    *(["--mask-chemical-channels"] if mask else []),
                ],
                out / "report.json",
            )
            with (c.root / "gpu.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                c.run_auxiliary(rid)
            c.evaluate_round(rid)
            c.complete_round(rid)
            models.append(out / "model.pt")
        reports = {}
        for name, model, rid in [
            ("chemical", models[0], "0255_query_strict_mass_chemical_system"),
            ("identity_masked", models[1], "0256_query_strict_mass_mask_system"),
        ]:
            source = complete_query_system(c, model, rid)
            if source is None:
                return
            report = json.loads((source / "report.json").read_text())
            reports[name] = {
                "round": source.name,
                "metrics": {
                    m: report[m]["chemical_query_residual"]
                    for m in ("unknown", "known")
                },
            }
        original_report = json.loads((original / "report.json").read_text())
        write_json(
            "results/research_loop/diagnostic_query_strict_mass_attribution.json",
            {
                "local_only": True,
                "independent_acceptance": False,
                "arms": reports,
                "chemical_minus_mask_unknown_mrr25": reports["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - reports["identity_masked"]["metrics"]["unknown"]["mrr25"],
                "strict_minus_unrestricted_chemical_unknown_mrr25": reports["chemical"][
                    "metrics"
                ]["unknown"]["mrr25"]
                - original_report["unknown"]["chemical_query_residual"]["mrr25"],
                "scope": "Matched fixed training rows/capacity/steps/gate; CE excludes pinned hard negatives outside observed35ppm/floor.006Da. No new negatives, candidates or labels for evaluation selection. Repeated complete development comparison, no fresh acceptance.",
            },
        )


if __name__ == "__main__":
    main()
