"""Activate complete bounded extended chemistry winners with matched zero control."""

import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.extended_candidate_compatibility import load_model
from casmi_ml.metfrag import digest


def activate(c, directory, result):
    winner = result.get("winner")
    if not winner:
        return None
    if not c.local_only():
        raise ValueError("Candidate compatibility activation is local-only")
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if (
        report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError(
            "Incomplete candidate compatibility evidence cannot be activated"
        )
    sources = {
        "source_sha256": "casmi_ml/extended_candidate_system.py",
        "inference_source_sha256": "casmi_ml/chembl_routed_inference.py",
        "compatibility_source_sha256": "casmi_ml/extended_candidate_compatibility.py",
        "evidence_source_sha256": "casmi_ml/diagnostic_residual.py",
        "extended_rule_source_sha256": "casmi_ml/extended_diagnostic_rules.py",
    }
    for field, source in sources.items():
        if protocol[field] != digest(source):
            raise ValueError(
                "Verified candidate compatibility source changed: " + field
            )
    zero = c.root / "rounds/0306_extended_candidate_zero_control"
    zero_protocol = json.loads((zero / "protocol.json").read_text())
    zero_report = json.loads((zero / "report.json").read_text())
    if not zero_report.get("control_passed") or not zero_protocol.get("control"):
        raise ValueError("Complete exact zero candidate bonus control required")
    for field in sources:
        if zero_protocol[field] != protocol[field]:
            raise ValueError(
                "Candidate compatibility zero control source differs: " + field
            )
    best = c.read()["development_best"]
    if best["directory"] != str(directory) or best["variant"] != winner["variant"]:
        raise ValueError(
            "Candidate compatibility activation must match current complete winner"
        )
    config = protocol["config"]
    model_path = protocol["model"]
    if digest(model_path) != protocol["model_sha256"]:
        raise ValueError("Verified candidate compatibility weights changed")
    _, model = load_model(model_path, config["encoder"], config["critic"])
    if model["mask_chemical_channels"]:
        raise ValueError(
            "Masked candidate bonus cannot be activated as chemical winner"
        )
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
            "model": model_path,
            "model_sha256": digest(model_path),
        },
        "metrics": winner["metrics"],
        "remote_publication": False,
        "independent_acceptance": False,
        "scope": "Complete repeated development candidate chemistry winner with full zero bonus control; not fresh holdout or cold resource acceptance",
    }
    if config.get("candidate_order_runtime") == "array_v1":
        proof["array_source_sha256"] = digest("casmi_ml/candidate_array_order.py")
    if config.get("structure_identity_runtime") == "cache_v1":
        identity_sha = config["structure_identity_source_sha256"]
        if identity_sha != digest("casmi_ml/cached_structure_keys.py"):
            raise ValueError("Candidate compatibility identity source changed")
        proof["identity_source_sha256"] = identity_sha
    write_json("results/research_loop/extended_candidate_activation.json", proof)
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
    if spec.get("variant") == "motif_count":
        from casmi_ml.candidate_count_activation import resolve_score as resolve_count

        return resolve_count(config, proof)
    if spec.get("variant") == "full_cosine_gap":
        from casmi_ml.full_gap_candidate_activation import (
            resolve_score as resolve_full_gap,
        )

        return resolve_full_gap(config, proof)
    if config.get("query_residual"):
        raise ValueError(
            "Candidate chemistry was evaluated over original frozen critic"
        )
    if digest(spec["model"]) != spec["model_sha256"]:
        raise ValueError("Active candidate compatibility weights changed")
    score, _ = load_model(spec["model"], config["encoder"], config["critic"])
    return score
