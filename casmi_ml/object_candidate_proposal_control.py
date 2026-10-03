"""Require exact three original proposal dictionaries with object candidate storage."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.chembl_prior_proposals import run as proposals
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.object_candidate_runtime import ObjectPackedCandidateIndex
from casmi_ml.proposal_runtime_control import ORIGINAL
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Object proposal control is local-only")
    output = Path(output)
    runtime = c.read()["local_candidate_runtime"]
    freeze(
        output / "object_protocol.json",
        {
            "source_sha256": digest(__file__),
            "object_source_sha256": digest("casmi_ml/object_candidate_runtime.py"),
            "candidate_runtime": runtime,
            "original_proposal_protocol_sha256": digest(ORIGINAL / "protocol.json"),
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch(
        "casmi_ml.chembl_prior_proposals.CandidateIndex",
        new=lambda catalog: ObjectPackedCandidateIndex(
            catalog, runtime["store"], runtime["catalog_sha256"]
        ),
    ):
        report = proposals(output, all_queries=True)
    matches = {}
    for name in ["native", "corrected_only", "corrected_fusion"]:
        actual = json.loads((output / (name + "_proposals.json")).read_text())
        expected = json.loads((ORIGINAL / (name + "_proposals.json")).read_text())
        if actual != expected:
            raise ValueError("Object candidate proposals differ: " + name)
        matches[name] = len(actual)
    report.update(
        local_only=True,
        runtime_verified=True,
        runtime_complete_proposal_matches=matches,
    )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
