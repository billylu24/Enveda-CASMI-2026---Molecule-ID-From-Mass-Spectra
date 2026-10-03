"""Explicit local proposal entry point guarded by an exact all-query cache control."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_prior_proposals import run as proposals
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_runtime import resolve as resolve_candidates
from casmi_ml.proposal_runtime_control import ORIGINAL
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, freeze

NAMES = ("native", "corrected_only", "corrected_fusion")


def validate_report(report):
    if not report.get("runtime_verified") or report.get(
        "runtime_complete_proposal_matches"
    ) != dict.fromkeys(NAMES, 2000):
        raise ValueError("Exact three complete2000-query proposal controls required")


def activate(c, directory):
    if not c.local_only():
        raise ValueError("Proposal runtime activation is local-only")
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    validate_report(report)
    protocol = json.loads((directory / "runtime_protocol.json").read_text())
    if protocol["source_sha256"] != digest("casmi_ml/proposal_runtime_control.py"):
        raise ValueError("Proposal control implementation changed")
    if protocol["candidate_runtime"] != c.read().get("local_candidate_runtime"):
        raise ValueError("Proposal control candidate cache binding changed")
    if protocol["original_proposal_protocol_sha256"] != digest(
        ORIGINAL / "protocol.json"
    ):
        raise ValueError("Original proposal protocol changed")
    for name in NAMES:
        if json.loads(
            (directory / (name + "_proposals.json")).read_text()
        ) != json.loads((ORIGINAL / (name + "_proposals.json")).read_text()):
            raise ValueError("Complete proposal dictionaries differ: " + name)
    if resolve_candidates(c, digest(DERIVED)) is None:
        raise ValueError("Verified packed candidate cache required")
    value = {
        "enabled": True,
        "local_only": True,
        "control_directory": str(directory),
        "report_sha256": digest(directory / "report.json"),
        "runtime_protocol_sha256": digest(directory / "runtime_protocol.json"),
        "proposal_source_sha256": digest("casmi_ml/chembl_prior_proposals.py"),
        "control_source_sha256": digest("casmi_ml/proposal_runtime_control.py"),
        "wrapper_source_sha256": digest(__file__),
        "original_protocol_sha256": digest(ORIGINAL / "protocol.json"),
        "candidate_runtime": c.read()["local_candidate_runtime"],
        "scope": "Explicit local full mass-window proposals only; three complete original-encoder dictionaries exact. Fingerprint runtime reuse, no accuracy improvement or cold inference acceptance.",
    }
    c.change(lambda state: state.update(local_proposal_runtime=value))
    write_json("results/research_loop/local_proposal_runtime_activation.json", value)
    return value


def verified_factory(c):
    if not c.local_only():
        raise ValueError("Local proposal runtime requires local-only mode")
    value = c.read().get("local_proposal_runtime", {})
    if not value.get("enabled"):
        raise ValueError("No exact proposal runtime control active")
    directory = Path(value["control_directory"])
    for path, field in [
        (directory / "report.json", "report_sha256"),
        (directory / "runtime_protocol.json", "runtime_protocol_sha256"),
        ("casmi_ml/chembl_prior_proposals.py", "proposal_source_sha256"),
        ("casmi_ml/proposal_runtime_control.py", "control_source_sha256"),
        (__file__, "wrapper_source_sha256"),
        (ORIGINAL / "protocol.json", "original_protocol_sha256"),
    ]:
        if digest(path) != value[field]:
            raise ValueError(
                "Local proposal runtime source/evidence changed: " + str(path)
            )
    if value["candidate_runtime"] != c.read().get("local_candidate_runtime"):
        raise ValueError("Proposal packed candidate cache changed")
    validate_report(json.loads((directory / "report.json").read_text()))
    factory = resolve_candidates(c, digest(DERIVED))
    if factory is None:
        raise ValueError("Verified packed candidate runtime required")
    return factory, value


def run(output, encoder=ENCODER):
    c = Controller()
    factory, proof = verified_factory(c)
    output = Path(output)
    freeze(
        output / "local_runtime_protocol.json",
        {
            "source_sha256": digest(__file__),
            "runtime_proof": proof,
            "encoder_sha256": digest(encoder),
            "all_queries": True,
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch("casmi_ml.chembl_prior_proposals.CandidateIndex", new=factory):
        report = proposals(output, all_queries=True, encoder=encoder)
    report.update(
        local_only=True,
        runtime_control=proof["control_directory"],
        accuracy_gain_claim=False,
    )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--encoder", default=str(ENCODER))
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
