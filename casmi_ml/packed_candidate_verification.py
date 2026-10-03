"""All3340 original physical-input ranks under the packed candidate index."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.low_leader_replay import run as replay
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_cache import PackedCandidateIndex
from casmi_ml.research_protocol import freeze

STORE = Path("artifacts/research_loop/rounds/0250_local_packed_candidate_preparation")
VERIFIED = Path("artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay")
BENCHMARK = Path(
    "artifacts/research_loop/rounds/0251_packed_candidate_window_benchmark"
)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((VERIFIED / "external_config.json").read_text())
    benchmark = json.loads((BENCHMARK / "report.json").read_text())
    if not benchmark["verified"]:
        raise ValueError("Matched exact window benchmark required")
    freeze(
        output / "packed_protocol.json",
        {
            "source_sha256": digest(__file__),
            "packed_source_sha256": digest("casmi_ml/packed_candidate_cache.py"),
            "store_report_sha256": digest(STORE / "report.json"),
            "catalog_sha256": config["catalog_sha256"],
            "benchmark_report_sha256": digest(BENCHMARK / "report.json"),
            "rule": "Patch only catalog candidate index to source-bound packed exact fingerprints; require unchanged all3340 complete label-free lowleader ranks versus independently derived control. Existing physical cached request evidence unchanged.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch(
        "casmi_ml.chembl_routed_inference.CandidateIndex",
        new=lambda catalog: PackedCandidateIndex(
            catalog, STORE, config["catalog_sha256"]
        ),
    ):
        report = replay(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
