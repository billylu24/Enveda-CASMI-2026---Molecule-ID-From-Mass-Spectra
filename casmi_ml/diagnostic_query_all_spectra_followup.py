"""Match query correction training to complete observed multi-ion spectrum groups."""

import fcntl
import json
from pathlib import Path

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.data import write_json
from casmi_ml.diagnostic_query_followup import complete_query_system
from casmi_ml.diagnostic_system_followup import wait_for_report
from casmi_ml.research_loop import Controller


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("All-spectra query cycle is local-only")
    with (c.root / "diagnostic_query_all_spectra_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if wait_for_complete_system(c, "0292_query_admission_mask_system") is None:
            return
        audit = "0265_training_query_input_contribution_audit"
        if not wait_for_report(c, audit):
            return
        control = c.read().get("local_query_zero_control")
        if (
            not control
            or not json.loads((Path(control) / "report.json").read_text())[
                "control_passed"
            ]
        ):
            raise RuntimeError("Current verified zero query control required")
        models = []
        for mask, rid in [
            (False, "0294_query_all_spectra_chemical"),
            (True, "0295_query_all_spectra_mask"),
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
                    "--all-spectra",
                    *(["--mask-chemical-channels"] if mask else []),
                ],
                out / "report.json",
            )
            with (c.root / "gpu.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                c.run_auxiliary(rid)
            report = json.loads((out / "report.json").read_text())
            if not report["original_critic_tensors_exact"] or not report["all_spectra"]:
                raise ValueError(
                    "All-spectra masked aggregation or frozen critic verification failed"
                )
            c.evaluate_round(rid)
            c.complete_round(rid)
            models.append(out / "model.pt")
        arms = {}
        for name, model, rid in [
            ("chemical", models[0], "0296_query_all_spectra_chemical_system"),
            ("identity_masked", models[1], "0297_query_all_spectra_mask_system"),
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
        previous = json.loads(
            (
                c.root / "rounds/0262_query_multispectrum_chemical_system/report.json"
            ).read_text()
        )
        write_json(
            "results/research_loop/query_all_spectra_attribution.json",
            {
                "local_only": True,
                "remote_publication": False,
                "independent_acceptance": False,
                "arms": arms,
                "chemical_minus_mask_unknown_mrr25": arms["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - arms["identity_masked"]["metrics"]["unknown"]["mrr25"],
                "all_spectra_minus_same_ion_pair_unknown_mrr25": arms["chemical"][
                    "metrics"
                ]["unknown"]["mrr25"]
                - previous["unknown"]["chemical_query_residual"]["mrr25"],
                "scope": "Same raw contextual correction head, labels, original unrestricted15 CE and3epochs235steps as260/261; train mean of all1-4 original measured rows per molecule across observed adduct/mode with zero-weight padding. Original encoder/critic frozen and inference graph unchanged, matched identity mask preserves raw strength gate. All214804 actual training rows now used versus fixed120K pair slots; row increase is the aggregation intervention, no extra optimizer updates. Repeated full3340 comparison, no fresh acceptance.",
            },
        )


if __name__ == "__main__":
    main()
