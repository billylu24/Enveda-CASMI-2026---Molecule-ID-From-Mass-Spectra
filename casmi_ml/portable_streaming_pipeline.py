"""Source-bound scientific236 with bounded streaming physical request discovery."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.portable_packed_pipeline import run as packed_run
from casmi_ml.streaming_fragment_prefetch import extend_streamed


def run(recipe, data_dir, coconut, output):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "streaming_fragment_prefetch"
    ]
    if (
        proof["source_sha256"] != digest("casmi_ml/streaming_fragment_prefetch.py")
        or proof["prefetch_source_sha256"]
        != digest("casmi_ml/portable_fragment_prefetch.py")
        or proof["full_rank_matches"] != {"unknown": 2000, "known": 1340}
    ):
        raise ValueError("Source-bound full3340 streaming request control required")
    with patch("casmi_ml.chembl_routed_inference.extend", new=extend_streamed):
        result = packed_run(recipe, data_dir, coconut, output)
    report_path = Path(str(output) + ".report.json")
    report = json.loads(report_path.read_text())
    report["portable_execution_runtime"] = {
        "physical_workers": 2,
        "candidate_threads_per_worker": 2,
        "shared_original_fragment_deadline": True,
        "streaming_source_sha256": proof["source_sha256"],
        "prefetch_source_sha256": proof["prefetch_source_sha256"],
        "scope": "Discover and start unchanged original requests concurrently; bounded2 original JVMs, same input dictionaries, model scores, timeouts and shared deadline. No packaged query-dependent predictions or physical evidence.",
    }
    write_json(report_path, report)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
