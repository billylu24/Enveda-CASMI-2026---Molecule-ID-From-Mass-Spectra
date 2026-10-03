"""Activate object frame storage only after complete inference and proposal controls."""

import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.object_candidate_runtime import ObjectPackedCandidateIndex


def validate_reports(full, proposals):
    if (
        not full.get("verified")
        or full.get("development_budget_fallbacks")
        or full.get("incremental_new_failure_contexts")
    ):
        raise ValueError("Complete exact object candidate inference required")
    if any(
        full[mode]["full_rank_matches"] != count
        for mode, count in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("All3340 object candidate ranks required")
    if not proposals.get("runtime_verified") or proposals.get(
        "runtime_complete_proposal_matches"
    ) != dict.fromkeys(["native", "corrected_only", "corrected_fusion"], 2000):
        raise ValueError("All three complete2000 object proposal dictionaries required")


def activate(c, directory, proposal_directory):
    if not c.local_only():
        raise ValueError("Object candidate activation is local-only")
    directory, proposal_directory = Path(directory), Path(proposal_directory)
    full = json.loads((directory / "report.json").read_text())
    proposal = json.loads((proposal_directory / "report.json").read_text())
    validate_reports(full, proposal)
    protocol = json.loads((directory / "protocol.json").read_text())
    proposal_protocol = json.loads(
        (proposal_directory / "object_protocol.json").read_text()
    )
    state = c.read()
    if state["local_verified_inference"] != protocol["incumbent_proof"]:
        raise ValueError(
            "Object inference incumbent changed; replay against current best"
        )
    for evidence in [protocol, proposal_protocol]:
        if evidence["candidate_runtime"] != state[
            "local_candidate_runtime"
        ] or evidence["object_source_sha256"] != digest(
            "casmi_ml/object_candidate_runtime.py"
        ):
            raise ValueError("Object candidate source/cache binding changed")
    if protocol["inference_sha256"] != digest("casmi_ml/chembl_routed_inference.py"):
        raise ValueError("Original inference changed before object activation")
    value = {
        "enabled": True,
        "local_only": True,
        "control_directory": str(directory),
        "proposal_directory": str(proposal_directory),
        "source_sha256": protocol["object_source_sha256"],
        "candidate_runtime": state["local_candidate_runtime"],
        "report_sha256": digest(directory / "report.json"),
        "protocol_sha256": digest(directory / "protocol.json"),
        "proposal_report_sha256": digest(proposal_directory / "report.json"),
        "proposal_protocol_sha256": digest(proposal_directory / "object_protocol.json"),
        "scope": "Every3340 best full rank and three full2000 original proposal dictionaries exact; storage runtime only, no accuracy or cold acceptance gain",
    }
    c.change(lambda state: state.update(local_object_candidate_runtime=value))
    write_json("results/research_loop/object_candidate_runtime_activation.json", value)
    return value


def resolve(c, base):
    value = c.read().get("local_object_candidate_runtime", {})
    if not value.get("enabled"):
        return None
    if not c.local_only() or value["candidate_runtime"] != base:
        raise ValueError("Object candidate local/cache binding differs")
    full, proposal = Path(value["control_directory"]), Path(value["proposal_directory"])
    for path, field in [
        ("casmi_ml/object_candidate_runtime.py", "source_sha256"),
        (full / "report.json", "report_sha256"),
        (full / "protocol.json", "protocol_sha256"),
        (proposal / "report.json", "proposal_report_sha256"),
        (proposal / "object_protocol.json", "proposal_protocol_sha256"),
    ]:
        if digest(path) != value[field]:
            raise ValueError("Object candidate evidence/source changed: " + str(path))
    validate_reports(
        json.loads((full / "report.json").read_text()),
        json.loads((proposal / "report.json").read_text()),
    )
    return lambda catalog: ObjectPackedCandidateIndex(
        catalog, base["store"], base["catalog_sha256"]
    )
