"""Activate only complete winner inference from the already evaluated query-residual path."""

import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest


def activate(c, directory, result):
    winner = result.get("winner")
    if not winner:
        return None
    if not c.local_only():
        raise ValueError("Query activation is local-only")
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if (
        report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Incomplete query evidence cannot be activated")
    for name, field in [
        ("casmi_ml/chembl_routed_inference.py", "inference_source_sha256"),
        ("casmi_ml/diagnostic_query_ranker.py", "query_source_sha256"),
        ("casmi_ml/diagnostic_query_residual.py", "query_residual_sha256"),
    ]:
        if digest(name) != protocol[field]:
            raise ValueError("Verified query inference source changed")
    if protocol.get("strength_transform_sha256") and protocol[
        "strength_transform_sha256"
    ] != digest("casmi_ml/diagnostic_strength_transform.py"):
        raise ValueError("Query strength transform source changed")
    best = c.read()["development_best"]
    if best["directory"] != str(directory) or best["variant"] != winner["variant"]:
        raise ValueError("Query activation must match the current complete winner")
    zero = Path(
        c.read().get(
            "local_query_zero_control",
            c.root / "rounds/0244_diagnostic_query_zero_control",
        )
    )
    control = json.loads((zero / "report.json").read_text())
    control_protocol = json.loads((zero / "protocol.json").read_text())
    if not control.get("control_passed") or any(
        control_protocol[field] != protocol[field]
        for field in [
            "inference_source_sha256",
            "query_source_sha256",
            "query_residual_sha256",
        ]
    ):
        raise ValueError("Exact query zero-residual full-ranking control required")
    if protocol.get("strength_transform_sha256") != control_protocol.get(
        "strength_transform_sha256"
    ):
        raise ValueError("Query strength transform zero control source differs")
    proof = {
        "scientific_round": best["round"],
        "evidence_round": directory.name,
        "local_only": True,
        "replay_report_sha256": digest(directory / "report.json"),
        "rankings_sha256": digest(directory / "selected_rankings.json"),
        "external_config_sha256": digest(directory / "external_config.json"),
        "inference_sha256": protocol["inference_source_sha256"],
        "low_leader_source_sha256": digest("casmi_ml/low_leader_inference.py"),
        "metrics": winner["metrics"],
        "remote_publication": False,
        "independent_acceptance": False,
        "scope": "Complete cached candidate-facing residual winner with paired zero-residual implementation control; no fresh holdout or cold full pipeline acceptance",
    }
    write_json("results/research_loop/query_verified_runtime_activation.json", proof)
    c.change(
        lambda state: state.update(
            local_verified_inference={
                **proof,
                "external_config": str(directory / "external_config.json"),
            }
        )
    )
    return proof
