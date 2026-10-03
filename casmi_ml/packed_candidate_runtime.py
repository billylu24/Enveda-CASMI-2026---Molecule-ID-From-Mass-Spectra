"""Verified local candidate cache selection; scientific output remains unchanged."""

import json
from pathlib import Path

from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_cache import PackedCandidateIndex


def resolve(c, catalog_sha256):
    value = c.read().get("local_candidate_runtime", {})
    if not c.local_only() or not value.get("enabled"):
        return None
    if value["catalog_sha256"] != catalog_sha256:
        raise ValueError("Active packed cache catalog differs")
    if value["source_sha256"] != digest("casmi_ml/packed_candidate_cache.py"):
        raise ValueError("Active packed candidate implementation changed")
    control = Path(value["control_directory"])
    if digest(control / "report.json") != value["control_report_sha256"]:
        raise ValueError("Active packed inference control changed")
    report = json.loads((control / "report.json").read_text())
    if (
        not report.get("verified")
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Active packed cache has no complete inference control")
    return lambda catalog: PackedCandidateIndex(catalog, value["store"], catalog_sha256)


def activate(c, directory, store):
    if not c.local_only():
        raise ValueError("Packed runtime activation is local-only")
    directory, store = Path(directory), Path(store)
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "packed_protocol.json").read_text())
    if (
        not report.get("verified")
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete exact packed-runtime inference required")
    if any(
        report[mode]["full_rank_matches"] != count
        for mode, count in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("All3340 complete rankings must match")
    if protocol["packed_source_sha256"] != digest(
        "casmi_ml/packed_candidate_cache.py"
    ) or protocol["store_report_sha256"] != digest(store / "report.json"):
        raise ValueError("Packed runtime source binding differs")
    value = {
        "enabled": True,
        "local_only": True,
        "store": str(store),
        "control_directory": str(directory),
        "catalog_sha256": protocol["catalog_sha256"],
        "source_sha256": protocol["packed_source_sha256"],
        "control_report_sha256": digest(directory / "report.json"),
        "scope": "Exact all3340 full-ranking local runtime cache; candidate bit values and ordering unchanged, no accuracy increase or cold runtime acceptance claim",
    }
    c.change(lambda state: state.update(local_candidate_runtime=value))
    from casmi_ml.data import write_json

    write_json("results/research_loop/packed_candidate_runtime_activation.json", value)
    return value
