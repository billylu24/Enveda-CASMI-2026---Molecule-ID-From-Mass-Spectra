"""Match training query aggregation to actual multi-spectrum candidate scoring."""

import fcntl
import json

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.data import write_json
from casmi_ml.diagnostic_query_followup import complete_query_system
from casmi_ml.diagnostic_system_followup import wait_for_report
from casmi_ml.research_loop import Controller


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Multi-spectrum query cycle is local-only")
    with (c.root / "diagnostic_query_multispectrum_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = "0259_query_array_zero_control"
        if not wait_for_report(c, control):
            return
        if not json.loads((c.root / "rounds" / control / "report.json").read_text())[
            "control_passed"
        ]:
            raise RuntimeError("Current array-source zero query control did not pass")
        c.evaluate_round(control)
        c.complete_round(control)
        c.change(
            lambda state: state.update(
                local_query_zero_control=str(c.root / "rounds" / control)
            )
        )
        if wait_for_complete_system(c, "0256_query_strict_mass_mask_system") is None:
            return
        models = []
        for mask, rid in [
            (False, "0260_query_multispectrum_chemical"),
            (True, "0261_query_multispectrum_mask_control"),
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
                    "--multispectrum",
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
        arms = {}
        for name, model, rid in [
            ("chemical", models[0], "0262_query_multispectrum_chemical_system"),
            ("identity_masked", models[1], "0263_query_multispectrum_mask_system"),
        ]:
            source = complete_query_system(c, model, rid)
            if source is None:
                return
            report = json.loads((source / "report.json").read_text())
            arms[name] = {
                "round": source.name,
                "metrics": {
                    m: report[m]["chemical_query_residual"]
                    for m in ("unknown", "known")
                },
            }
        original = json.loads(
            (
                c.root / "rounds/0247_diagnostic_query_chemical_system/report.json"
            ).read_text()
        )
        write_json(
            "results/research_loop/query_multispectrum_attribution.json",
            {
                "local_only": True,
                "independent_acceptance": False,
                "arms": arms,
                "chemical_minus_mask_unknown_mrr25": arms["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - arms["identity_masked"]["metrics"]["unknown"]["mrr25"],
                "paired_minus_single_chemical_unknown_mrr25": arms["chemical"][
                    "metrics"
                ]["unknown"]["mrr25"]
                - original["unknown"]["chemical_query_residual"]["mrr25"],
                "scope": "Same original60K labels, fixed same-adduct/mode measured pair RNG260 after original representative RNG42, optimizer steps/capacity fixed; corrected per-spectrum queries averaged exactly as inference. Unrestricted pinned15 negative objective retained after strict-mass negative rejection. Repeated full3340 comparison, no fresh acceptance.",
            },
        )


if __name__ == "__main__":
    main()
