"""Sequential optional peak-loss attenuation with a new source-bound zero control."""

import fcntl
import importlib
import json
import time
from pathlib import Path

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.data import write_json
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.local_research_snapshot import refresh
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller

TRIAL = Path("artifacts/research_loop/strong_peak_source_trial/amendment.json")


def install_trial():
    amendments = json.loads(TRIAL.read_text())
    for path, value in amendments.items():
        if digest(value["trial"]) != value["trial_sha256"] or digest(path) not in [
            value["original_sha256"],
            value["trial_sha256"],
        ]:
            raise ValueError("Strong-peak source amendment changed: " + path)
    for path, value in amendments.items():
        if digest(path) != value["trial_sha256"]:
            Path(path).write_bytes(Path(value["trial"]).read_bytes())
    # Ensure this driver's imports follow the now installed source, rather than
    # retaining any old class captured by an indirectly imported helper.
    for module in [
        "casmi_ml.diagnostic_query_residual",
        "casmi_ml.diagnostic_query_ranker",
        "casmi_ml.diagnostic_query_followup",
    ]:
        importlib.reload(importlib.import_module(module))
    write_json(
        "results/research_loop/query_strong_peak_source_amendment.json",
        {
            "local_only": True,
            "sources": amendments,
            "scope": "Optional false-by-default transform; previous scientific arms finalized before source update, incumbent original critic graph unchanged. New zero280 required for any transformed query acceptance.",
        },
    )


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Strong-peak query followup is local-only")
    with (c.root / "diagnostic_query_strong_peak_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if wait_for_complete_system(c, "0275_query_balanced_mask_system") is None:
            return
        while not c.stopped() and not c.read().get(
            "local_object_candidate_runtime", {}
        ).get("enabled"):
            time.sleep(10)
        if c.stopped():
            return
        install_trial()
        from casmi_ml.diagnostic_query_followup import (
            complete_query_system,
            prepare_zero,
        )

        zero = prepare_zero(
            c.root / "rounds/0280_query_strong_peak_zero_control",
            strong_peak_losses=True,
        )
        source, _ = execute(
            c,
            "0280_query_strong_peak_zero_control",
            "casmi_ml.diagnostic_query_system",
            ["--critic", str(zero), "--control"],
            "implementation_control",
        )
        report = json.loads((source / "report.json").read_text())
        if (
            not report["control_passed"]
            or report["development_budget_fallbacks"]
            or report["incremental_new_failure_contexts"]
        ):
            raise ValueError("Strong-peak exact full3340 zero control failed")
        c.change(lambda state: state.update(local_query_zero_control=str(source)))
        models = []
        for mask, rid in [
            (False, "0281_query_strong_peak_chemical"),
            (True, "0282_query_strong_peak_mask"),
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
                    "--strong-peak-losses",
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
            ("chemical", models[0], "0283_query_strong_peak_chemical_system"),
            ("identity_masked", models[1], "0284_query_strong_peak_mask_system"),
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
                c.root / "rounds/0262_query_multispectrum_chemical_system/report.json"
            ).read_text()
        )
        write_json(
            "results/research_loop/query_strong_peak_attribution.json",
            {
                "local_only": True,
                "remote_publication": False,
                "independent_acceptance": False,
                "arms": arms,
                "chemical_minus_mask_unknown_mrr25": arms["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - arms["identity_masked"]["metrics"]["unknown"]["mrr25"],
                "strong_minus_raw_chemical_unknown_mrr25": arms["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - original["unknown"]["chemical_query_residual"]["mrr25"],
                "scope": "Same original paired contextual training as260/261, only peak-loss feature identity transform relu(x-.3)/.7; raw evidence maximum-strength gate preserved in both arms, ion/precursor-loss channels untouched. Training-only279 motivated fixed transform, no development threshold fitting; repeated full3340 evaluation, no independent acceptance.",
            },
        )
        refresh()


if __name__ == "__main__":
    main()
