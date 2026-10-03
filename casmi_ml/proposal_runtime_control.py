"""Packed candidate runtime control for all-query external diagnostics."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_prior_proposals import run as proposals
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_runtime import resolve
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

ORIGINAL = Path("artifacts/research_loop/rounds/0147_chembl_all_query_proposals")


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Proposal runtime control is local-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    factory = resolve(c, digest(DERIVED))
    if factory is None:
        raise ValueError("Verified packed candidate runtime required")
    freeze(
        output / "runtime_protocol.json",
        {
            "source_sha256": digest(__file__),
            "candidate_runtime": c.read()["local_candidate_runtime"],
            "original_proposal_protocol_sha256": digest(ORIGINAL / "protocol.json"),
            "rule": "Original encoder and every complete mass window; packed candidate fingerprints only. Require exact complete native/corrected/fusion ranking dictionaries against original147 before treating this runtime as valid external diagnostics.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch("casmi_ml.chembl_prior_proposals.CandidateIndex", new=factory):
        report = proposals(output, all_queries=True)
    matches = {}
    for name in ["native", "corrected_only", "corrected_fusion"]:
        expected = json.loads((ORIGINAL / (name + "_proposals.json")).read_text())
        actual = json.loads((output / (name + "_proposals.json")).read_text())
        if actual != expected:
            raise ValueError("Packed external full proposals differ: " + name)
        matches[name] = len(actual)
    report.update(
        local_only=True,
        runtime_complete_proposal_matches=matches,
        runtime_verified=True,
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
