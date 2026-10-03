"""Resolve only the complete source-bound array sort runtime amendment."""

import json
from pathlib import Path

from casmi_ml.metfrag import digest


def resolve(c):
    value = c.read().get("local_array_runtime", {})
    if not c.local_only() or not value.get("enabled"):
        return None
    source = Path(value["control_directory"])
    report = json.loads((source / "report.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    if (
        not report["verified"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Array runtime has no complete exact rank control")
    if (
        digest("casmi_ml/chembl_routed_inference.py")
        != protocol["source_amendment"]["amended_inference_sha256"]
        or digest("casmi_ml/candidate_array_order.py")
        != protocol["array_source_sha256"]
    ):
        raise ValueError("Array runtime source changed")
    if any(
        report[mode]["full_rank_matches"] != count
        for mode, count in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("Array runtime full cohort ranks incomplete")
    return "array_v1"
