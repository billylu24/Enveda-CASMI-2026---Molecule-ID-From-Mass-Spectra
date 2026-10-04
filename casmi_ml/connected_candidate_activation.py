"""Activate a qualified connected-ion scorer only after complete matching controls."""

import json
from pathlib import Path

from casmi_ml.connected_candidate_compatibility import load_model
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest

SOURCES = {
    "source_sha256": "casmi_ml/connected_candidate_system.py",
    "base_system_source_sha256": "casmi_ml/full_gap_candidate_system.py",
    "inference_source_sha256": "casmi_ml/chembl_routed_inference.py",
    "compatibility_source_sha256": "casmi_ml/connected_candidate_compatibility.py",
    "original_compatibility_source_sha256": "casmi_ml/full_gap_candidate_compatibility.py",
    "connected_source_sha256": "casmi_ml/connected_fragment_audit.py",
    "refinement_source_sha256": "casmi_ml/connected_fragment_capacity_audit.py",
    "evidence_source_sha256": "casmi_ml/diagnostic_residual.py",
    "extended_rule_source_sha256": "casmi_ml/extended_diagnostic_rules.py",
}


def activate(c, directory, result):
    winner = result.get("winner")
    if not winner:
        return None
    if not c.local_only() or not winner["gate"]["eligible"]:
        raise ValueError("Local qualified connected winner required")
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if (
        report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete physical evidence required")
    for field, source in SOURCES.items():
        if protocol[field] != digest(source):
            raise ValueError("Connected activation source changed: " + field)
    for rid in (
        "0341_connected_candidate_zero_control",
        "0345_connected_candidate_mask_system",
    ):
        control = c.root / "rounds" / rid
        control_report = json.loads((control / "report.json").read_text())
        control_protocol = json.loads((control / "protocol.json").read_text())
        if not control_report["control_passed"] or not control_protocol["control"]:
            raise ValueError("All3340 exact zero and mask controls required")
        for field in SOURCES:
            if control_protocol[field] != protocol[field]:
                raise ValueError("Connected control source differs")
    best = c.read()["development_best"]
    if best["directory"] != str(directory) or best["variant"] != winner["variant"]:
        raise ValueError("Connected activation must match the current complete winner")
    config = protocol["config"]
    model = protocol["model"]
    if digest(model) != protocol["model_sha256"]:
        raise ValueError("Connected model changed")
    _, weights = load_model(model, config["encoder"], config["critic"])
    if weights["mask_chemical_channels"]:
        raise ValueError("Masked scorer cannot replace chemical winner")
    proof = {
        "scientific_round": best["round"],
        "evidence_round": directory.name,
        "local_only": True,
        "replay_report_sha256": digest(directory / "report.json"),
        "rankings_sha256": digest(directory / "selected_rankings.json"),
        "external_config_sha256": digest(directory / "external_config.json"),
        "inference_sha256": protocol["inference_source_sha256"],
        "low_leader_source_sha256": digest("casmi_ml/low_leader_inference.py"),
        "extended_candidate_compatibility": {
            "variant": "connected_ion",
            "model": model,
            "model_sha256": digest(model),
        },
        "metrics": winner["metrics"],
        "remote_publication": False,
        "independent_acceptance": False,
        "scope": "Complete repeated-development connected winner, full matching zero/mask controls. No cold or independent accuracy acceptance.",
    }
    if config.get("candidate_order_runtime") == "array_v1":
        proof["array_source_sha256"] = digest("casmi_ml/candidate_array_order.py")
    if config.get("structure_identity_runtime") == "cache_v1":
        proof["identity_source_sha256"] = config["structure_identity_source_sha256"]
    write_json("results/research_loop/candidate_connected_activation.json", proof)
    c.change(
        lambda state: state.update(
            local_verified_inference={
                **proof,
                "external_config": str(directory / "external_config.json"),
            }
        )
    )
    return proof


def resolve_score(config, proof):
    spec = proof.get("extended_candidate_compatibility")
    if not spec:
        return None
    if spec.get("variant") != "connected_ion" or config.get("query_residual"):
        raise ValueError("Bound original-critic connected-ion variant required")
    if digest(spec["model"]) != spec["model_sha256"]:
        raise ValueError("Active connected model changed")
    return load_model(spec["model"], config["encoder"], config["critic"])[0]
