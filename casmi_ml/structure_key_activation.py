"""Apply a source-bound identity cache only after current-best exact full replay."""

import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest


def activate(c, directory):
    if not c.local_only():
        raise ValueError("Identity cache activation is local-only")
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    proof = c.read()["local_verified_inference"]
    if (
        not report.get("verified")
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
        or proof != protocol["incumbent_proof"]
    ):
        raise ValueError("Complete unchanged-incumbent identity replay required")
    if any(
        report[mode]["full_rank_matches"] != count
        for mode, count in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("All3340 full identity ranks required")
    amendment = protocol["source_amendment"]
    if (
        digest("casmi_ml/chembl_routed_inference.py")
        != amendment["original_inference_sha256"]
        or digest(amendment["amended_source"]) != amendment["amended_inference_sha256"]
        or digest("casmi_ml/cached_structure_keys.py")
        != protocol["identity_source_sha256"]
    ):
        raise ValueError("Identity original/trial/cache source binding changed")
    if digest("casmi_ml/candidate_array_order.py") != protocol["array_source_sha256"]:
        raise ValueError("Original array sort source changed")
    Path("casmi_ml/chembl_routed_inference.py").write_bytes(
        Path(amendment["amended_source"]).read_bytes()
    )
    new = {
        **proof,
        "evidence_round": directory.name,
        "replay_report_sha256": digest(directory / "report.json"),
        "rankings_sha256": digest(directory / "selected_rankings.json"),
        "external_config_sha256": digest(directory / "external_config.json"),
        "inference_sha256": amendment["amended_inference_sha256"],
        "identity_source_sha256": protocol["identity_source_sha256"],
        "scope": "Same current scientific best every3340 full rank and all models/scores/evidence; bounded actualSMILES identity runtime cache only, no accuracy gain or cold acceptance",
    }
    c.change(
        lambda state: state.update(
            local_verified_inference=new,
            local_array_runtime={
                "enabled": True,
                "local_only": True,
                "source_sha256": protocol["array_source_sha256"],
                "control_directory": str(directory),
            },
        )
    )
    write_json("results/research_loop/structure_key_runtime_activation.json", new)
    return new
