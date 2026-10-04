"""Portable original scientific236 with source-bound two-JVM request scheduling."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.portable_fragment_prefetch import extend_prefetched
from casmi_ml.portable_packed_pipeline import run as packed_run


def run(recipe, data_dir, coconut, output):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "portable_fragment_prefetch"
    ]
    if (
        proof["source_sha256"] != digest("casmi_ml/portable_fragment_prefetch.py")
        or proof["full_rank_matches"] != {"unknown": 2000, "known": 1340}
        or proof["physical_workers"] != 2
    ):
        raise ValueError("Source-bound complete parallel fragment control required")
    with patch("casmi_ml.chembl_routed_inference.extend", new=extend_prefetched):
        result = packed_run(recipe, data_dir, coconut, output)
    report_path = Path(str(output) + ".report.json")
    report = json.loads(report_path.read_text())
    report["portable_execution_runtime"] = {
        "physical_workers": 2,
        "candidate_threads_per_worker": 2,
        "shared_original_fragment_deadline": True,
        "prefetch_source_sha256": proof["source_sha256"],
        "scope": "Unchanged scientific recipe/models and original request dictionaries; duplicate original critic/probability calls reused within this invocation only. Independent original JVMs retain original timeout/deadline. No packaged query-dependent scores/cache.",
    }
    write_json(report_path, report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
