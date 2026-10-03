"""Matched candidate-facing chemical and identity-masked query critic cycle."""

import fcntl
import json

import torch

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.data import write_json
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.diagnostic_query_activation import activate
from casmi_ml.diagnostic_query_ranker import DiagnosticQueryRanker
from casmi_ml.direct_models import DirectRanker
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller


def prepare_zero(directory, strong_peak_losses=False):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    weights = torch.load(CRITIC, map_location="cpu", weights_only=True)
    base = DirectRanker("fingerprint")
    base.load_state_dict(weights["state_dict"])
    model = DiagnosticQueryRanker(base, strong_peak_losses=strong_peak_losses)
    weights["state_dict"] = model.state_dict()
    weights["diagnostic_query_residual"] = {
        "base_critic_sha256": digest(CRITIC),
        "dimension": 24,
        "precise": True,
        "mask_chemical_channels": False,
        "strong_peak_losses": strong_peak_losses,
        "strength_transform_sha256": digest(
            "casmi_ml/diagnostic_strength_transform.py"
        ),
    }
    path = directory / "zero_model.pt"
    if not path.exists():
        torch.save(weights, path)
    return path


def complete_query_system(c, model, rid):
    for attempt in range(3):
        if c.stopped():
            return None
        trial = rid if attempt == 0 else rid + f"_cache_completion{attempt}"
        source, result = execute(
            c,
            trial,
            "casmi_ml.diagnostic_query_system",
            ["--critic", str(model)],
            "chemical_query_critic",
        )
        report = json.loads((source / "report.json").read_text())
        if (
            not report["development_budget_fallbacks"]
            and not report["incremental_new_failure_contexts"]
        ):
            activate(c, source, result)
            print(
                "Complete local query system",
                trial,
                "winner",
                result.get("winner"),
                flush=True,
            )
            return source
    raise RuntimeError(
        "Query residual bounded physical completion retries exhausted: " + rid
    )


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Query critic cycle is local-only")
    with (c.root / "diagnostic_query_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Complete both current precision arms before changing the ranking
        # objective; GPU follows peak-mask control to avoid overlap.
        if wait_for_complete_system(c, "0230_diagnostic_context_exact_system") is None:
            return
        zero = prepare_zero(c.root / "rounds/0244_diagnostic_query_zero_control")
        source, _ = execute(
            c,
            "0244_diagnostic_query_zero_control",
            "casmi_ml.diagnostic_query_system",
            ["--critic", str(zero), "--control"],
            "implementation_control",
        )
        if not json.loads((source / "report.json").read_text())["control_passed"]:
            raise RuntimeError("Zero query residual control failed")
        if (
            wait_for_complete_system(c, "0243_diagnostic_context_precursor_system")
            is None
        ):
            return
        models = []
        for mask, rid in [
            (False, "0245_diagnostic_query_chemical"),
            (True, "0246_diagnostic_query_mask_control"),
        ]:
            if c.stopped():
                return
            output = c.root / "rounds" / rid
            c.register_round(
                rid,
                "representation_diagnostic",
                [
                    ".venv-gpu/bin/python",
                    "-u",
                    "-m",
                    "casmi_ml.diagnostic_query_training",
                    "--output",
                    str(output),
                    *(["--mask-chemical-channels"] if mask else []),
                ],
                output / "report.json",
            )
            with (c.root / "gpu.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                c.run_auxiliary(rid)
            c.evaluate_round(rid)
            c.complete_round(rid)
            models.append(output / "model.pt")
        arms = {}
        for name, model, rid in [
            ("chemical", models[0], "0247_diagnostic_query_chemical_system"),
            ("identity_masked", models[1], "0248_diagnostic_query_mask_system"),
        ]:
            source = complete_query_system(c, model, rid)
            if source is None:
                return
            report = json.loads((source / "report.json").read_text())
            arms[name] = {
                "round": source.name,
                "metrics": {
                    mode: report[mode]["chemical_query_residual"]
                    for mode in ("unknown", "known")
                },
            }
        write_json(
            "results/research_loop/diagnostic_query_attribution.json",
            {
                "local_only": True,
                "independent_acceptance": False,
                "arms": arms,
                "chemical_minus_mask_unknown_mrr25": arms["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - arms["identity_masked"]["metrics"]["unknown"]["mrr25"],
                "scope": "Matched candidate-facing query residual, all original encoder and critic tensor values frozen, same capacity/training rows/hard negatives/steps/presence gate. Repeated full-development attribution, not fresh acceptance or cold inference proof.",
            },
        )


if __name__ == "__main__":
    main()
