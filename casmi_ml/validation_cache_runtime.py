"""Activate source-bound pure SMILES validation caching after complete local controls."""

import json
from pathlib import Path

from casmi_ml.cached_submission_validation import validate_submission
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest


def activate(c, directory, benchmark):
    if not c.local_only():
        raise ValueError("Validation cache activation is local-only")
    directory, benchmark = Path(directory), Path(benchmark)
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    bench = json.loads((benchmark / "report.json").read_text())
    if (
        not report.get("verified")
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete validation-cache ranking evidence required")
    if any(
        report[m]["full_rank_matches"] != n
        for m, n in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("All3340 validation-cache full rankings required")
    if not bench["verified_all_actual_input_and_output_rows"]:
        raise ValueError("Every actual validation input/output row required")
    times = bench["timings"]
    legacy = sum(x["seconds"] for x in times if x["implementation"] == "legacy")
    cached = sum(x["seconds"] for x in times if x["implementation"] == "cache")
    if cached >= 0.9 * legacy:
        raise ValueError("Actual validation-only speed improvement required")
    for field, path in {
        "source_sha256": "casmi_ml/validation_cache_full_control.py",
        "entry_source_sha256": "casmi_ml/local_verified_extension.py",
        "cache_source_sha256": "casmi_ml/cached_submission_validation.py",
        "original_validator_source_sha256": "casmi_ml/inference.py",
        "benchmark_report_sha256": benchmark / "report.json",
    }.items():
        if protocol[field] != digest(path):
            raise ValueError("Validation-cache control source differs: " + field)
    if c.read()["local_verified_inference"] != protocol["baseline_proof"]:
        raise ValueError(
            "Validation-cache control must match current complete scientific/runtime proof"
        )
    value = {
        "enabled": True,
        "local_only": True,
        "scientific_round": protocol["baseline_proof"]["scientific_round"],
        "control_directory": str(directory),
        "control_report_sha256": digest(directory / "report.json"),
        "control_protocol_sha256": digest(directory / "protocol.json"),
        "cache_source_sha256": protocol["cache_source_sha256"],
        "original_validator_source_sha256": protocol[
            "original_validator_source_sha256"
        ],
        "entry_source_sha256": protocol["entry_source_sha256"],
        "runtime_source_sha256": digest(__file__),
        "scope": "Pure SMILES validity/identity memoization with all3340 cached inference full ranks exact and validation-only paired timing; no accuracy or cold pipeline gain claim",
    }
    c.change(lambda state: state.update(local_validation_runtime=value))
    write_json("results/research_loop/validation_cache_runtime_activation.json", value)
    return value


def resolve(c):
    value = c.read().get("local_validation_runtime", {})
    if not c.local_only() or not value.get("enabled"):
        return None
    if value["scientific_round"] != c.read()["development_best"]["round"]:
        return None
    sources = {
        "runtime_source_sha256": __file__,
        "cache_source_sha256": "casmi_ml/cached_submission_validation.py",
        "original_validator_source_sha256": "casmi_ml/inference.py",
        "entry_source_sha256": "casmi_ml/local_verified_extension.py",
    }
    for field, path in sources.items():
        if value[field] != digest(path):
            raise ValueError("Active validation-cache source changed: " + field)
    directory = Path(value["control_directory"])
    if (
        digest(directory / "report.json") != value["control_report_sha256"]
        or digest(directory / "protocol.json") != value["control_protocol_sha256"]
    ):
        raise ValueError("Active validation-cache full control changed")
    report = json.loads((directory / "report.json").read_text())
    if (
        not report["verified"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Active validation cache has incomplete evidence")
    return validate_submission
