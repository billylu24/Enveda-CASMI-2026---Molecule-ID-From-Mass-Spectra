"""Enable a source-bound local BLAS concurrency setting after complete rank controls."""

import json
from contextlib import nullcontext
from pathlib import Path

from threadpoolctl import threadpool_info, threadpool_limits

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest


def activate(c, directory, benchmark):
    if not c.local_only():
        raise ValueError("BLAS runtime activation is local-only")
    directory, benchmark = Path(directory), Path(benchmark)
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    bench = json.loads((benchmark / "report.json").read_text())
    if (
        not report.get("verified")
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete BLAS rank evidence required")
    if any(
        report[m]["full_rank_matches"] != n
        for m, n in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("All3340 BLAS full rankings must match")
    if (
        bench["windows"] != 200
        or bench["candidate_ranking_mismatches"]
        or bench["score_seconds_two_repetitions"]["blas1"]
        >= 0.9 * bench["score_seconds_two_repetitions"]["blas24"]
    ):
        raise ValueError("Complete improved actual BLAS score microbenchmark required")
    if (
        protocol["entry_source_sha256"]
        != digest("casmi_ml/local_verified_extension.py")
        or protocol["source_sha256"] != digest("casmi_ml/blas_runtime_control.py")
        or protocol["benchmark_report_sha256"] != digest(benchmark / "report.json")
    ):
        raise ValueError("BLAS control source binding differs")
    if (
        c.read()["development_best"]["round"]
        != protocol["baseline_proof"]["scientific_round"]
    ):
        raise ValueError("BLAS control must bind current scientific best")
    value = {
        "enabled": True,
        "local_only": True,
        "blas_threads": 1,
        "scientific_round": protocol["baseline_proof"]["scientific_round"],
        "control_directory": str(directory),
        "control_report_sha256": digest(directory / "report.json"),
        "control_protocol_sha256": digest(directory / "protocol.json"),
        "benchmark_directory": str(benchmark),
        "benchmark_report_sha256": digest(benchmark / "report.json"),
        "runtime_source_sha256": digest(__file__),
        "entry_source_sha256": digest("casmi_ml/local_verified_extension.py"),
        "scope": "Full cached3340 ranks exact under BLAS1; actual candidate score-segment microbenchmark only, no scientific accuracy or cold pipeline gain claim",
    }
    c.change(lambda state: state.update(local_blas_runtime=value))
    write_json("results/research_loop/blas_runtime_activation.json", value)
    return value


def resolve(c):
    value = c.read().get("local_blas_runtime", {})
    if not c.local_only() or not value.get("enabled"):
        return nullcontext()
    if value["scientific_round"] != c.read()["development_best"]["round"]:
        return nullcontext()
    if (
        value["blas_threads"] != 1
        or value["runtime_source_sha256"] != digest(__file__)
        or value["entry_source_sha256"]
        != digest("casmi_ml/local_verified_extension.py")
    ):
        raise ValueError("Active BLAS runtime source changed")
    directory = Path(value["control_directory"])
    benchmark = Path(value["benchmark_directory"])
    for path, field in [
        (directory / "report.json", "control_report_sha256"),
        (directory / "protocol.json", "control_protocol_sha256"),
        (benchmark / "report.json", "benchmark_report_sha256"),
    ]:
        if digest(path) != value[field]:
            raise ValueError("Active BLAS controlled evidence changed")
    report = json.loads((directory / "report.json").read_text())
    if (
        not report["verified"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Active BLAS runtime has incomplete evidence")
    expected = json.loads((benchmark / "protocol.json").read_text())["blas_info"]
    actual = [x for x in threadpool_info() if x["user_api"] == "blas"]
    identity = lambda x: (
        x["internal_api"],
        x["prefix"],
        x["version"],
        x["architecture"],
        x["filepath"],
    )
    if sorted(map(identity, actual)) != sorted(map(identity, expected)):
        raise ValueError("Active BLAS library identity changed; repeat actual controls")
    return threadpool_limits(limits=1, user_api="blas")
