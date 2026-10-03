"""Local matched contextual diagnostics after completed evidence-only model evaluations."""

import fcntl
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.diagnostic_system_followup import wait_for_report
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller


def activate_local_runtime(c):
    control = c.root / "rounds/0221_incremental_failure_context_control"
    benchmark = c.root / "rounds/0215_incremental_fragment_fresh_benchmark"
    adapter = c.root / "rounds/0222_incremental_adapter_fresh_verification"
    report = json.loads((control / "report.json").read_text())
    protocol = json.loads((control / "protocol.json").read_text())
    verification = json.loads((adapter / "report.json").read_text())
    verification_protocol = json.loads((adapter / "protocol.json").read_text())
    source = digest("casmi_ml/development_incremental_metfrag.py")
    if (
        not c.local_only()
        or not protocol.get("control")
        or report["development_budget_fallbacks"]
        or protocol["development_incremental_source_sha256"] != source
        or not verification.get("verified")
        or verification_protocol["adapter_source_sha256"] != source
    ):
        raise ValueError(
            "Local incremental runtime has no matching complete control and fresh adapter verification"
        )
    # The strict evaluation already required every full ranking to equal the
    # independently reconstructed all-physical reference, before writing report.
    failure_file = control / "private_incremental_failures.json"
    runtime = {
        "enabled": True,
        "local_only": True,
        "control_directory": str(control),
        "benchmark_directory": str(benchmark),
        "adapter_verification_directory": str(adapter),
        "control_report_sha256": digest(control / "report.json"),
        "control_rankings_sha256": digest(control / "selected_rankings.json"),
        "failures_sha256": digest(failure_file),
        "adapter_source_sha256": source,
        "rule": "Same verified raw per-candidate physical evidence; only exact inherited original-control failure hashes allowed; no new failed or partial group promotion.",
    }
    c.change(lambda s: s.update(local_development_incremental=runtime))
    write_json(
        "results/research_loop/local_incremental_runtime_activation.json",
        {
            "local_only": True,
            "control": control.name,
            "fresh_verification": adapter.name,
            "source_sha256": source,
            "inherited_failure_contexts": len(json.loads(failure_file.read_text())),
            "remote_publication": False,
        },
    )


def complete_system(c, model, proposals, rid):
    last = None
    for attempt in range(3):
        if c.stopped():
            return None
        trial = rid if attempt == 0 else rid + f"_cache_completion{attempt}"
        out, result = execute(
            c,
            trial,
            "casmi_ml.diagnostic_system_evaluation",
            [
                "--encoder",
                str(c.root / "rounds" / model / "model.pt"),
                "--proposals",
                str(c.root / "rounds" / proposals / "corrected_only_proposals.json"),
                "--paired-control",
                str(c.root / "rounds/0206_full181_audited_exact_control"),
            ],
            "diagnostic_residual_system",
        )
        report = json.loads((out / "report.json").read_text())
        last = out
        if not report.get("development_budget_fallbacks") and not report.get(
            "incremental_new_failure_contexts"
        ):
            print(
                "Complete local system result",
                trial,
                "winner",
                result.get("winner"),
                flush=True,
            )
            return out
        print(
            "Cached physical evidence incomplete",
            trial,
            "budget",
            report.get("development_budget_fallbacks"),
            "new failures",
            report.get("incremental_new_failure_contexts"),
            flush=True,
        )
    raise RuntimeError(
        "Local bounded retries still have incomplete evidence: " + str(last)
    )


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("This cycle is local-only")
    with (c.root / "diagnostic_next_cycle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for rid in (
            "0221_incremental_failure_context_control",
            "0222_incremental_adapter_fresh_verification",
        ):
            if not wait_for_report(c, rid):
                return
        activate_local_runtime(c)
        for rid in ("0207_diagnostic_coarse_system", "0208_diagnostic_exact_system"):
            if not wait_for_report(c, rid):
                return
        complete_system(
            c,
            "0186_diagnostic_residual_coarse",
            "0188_diagnostic_coarse_proposals",
            "0223_diagnostic_coarse_complete_system",
        )
        complete_system(
            c,
            "0187_diagnostic_residual_exact",
            "0189_diagnostic_exact_proposals",
            "0224_diagnostic_exact_complete_system",
        )
        for precise, train, proposals in [
            (
                False,
                "0225_diagnostic_context_coarse",
                "0227_diagnostic_context_coarse_proposals",
            ),
            (
                True,
                "0226_diagnostic_context_exact",
                "0228_diagnostic_context_exact_proposals",
            ),
        ]:
            if c.stopped():
                return
            args = ["--contextual"] + (["--precise"] if precise else [])
            out = c.root / "rounds" / train
            c.register_round(
                train,
                "representation_diagnostic",
                [
                    ".venv-gpu/bin/python",
                    "-u",
                    "-m",
                    "casmi_ml.diagnostic_residual_training",
                    "--output",
                    str(out),
                    *args,
                ],
                out / "report.json",
            )
            with (c.root / "gpu.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                c.run_auxiliary(train)
            c.evaluate_round(train)
            c.complete_round(train)
            execute(
                c,
                proposals,
                "casmi_ml.chembl_prior_proposals",
                ["--all-queries", "--encoder", str(out / "model.pt")],
                "candidate_diagnostic",
            )
        complete_system(
            c,
            "0225_diagnostic_context_coarse",
            "0227_diagnostic_context_coarse_proposals",
            "0229_diagnostic_context_coarse_system",
        )
        complete_system(
            c,
            "0226_diagnostic_context_exact",
            "0228_diagnostic_context_exact_proposals",
            "0230_diagnostic_context_exact_system",
        )
        write_json(
            Path("results/research_loop/diagnostic_context_cycle.json"),
            {
                "status": "complete",
                "local_only": True,
                "independent_acceptance": False,
                "remote_publication": False,
                "evidence_only_complete": [
                    "0223_diagnostic_coarse_complete_system",
                    "0224_diagnostic_exact_complete_system",
                ],
                "contextual_systems": [
                    "0229_diagnostic_context_coarse_system",
                    "0230_diagnostic_context_exact_system",
                ],
            },
        )


if __name__ == "__main__":
    main()
