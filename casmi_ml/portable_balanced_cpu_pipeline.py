"""Frozen236 dualGPU pipeline with source-bound balanced external CPU/catalog reuse."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.balanced_cpu_fragment_runtime import extend_balanced
from casmi_ml.metfrag import digest
from casmi_ml.portable_parallel_generation_pipeline import run as original_run


def run(recipe, data_dir, coconut, output, force_partitions=False):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "balanced_cpu_runtime"
    ]
    if (
        proof["runtime_source_sha256"]
        != digest("casmi_ml/balanced_cpu_fragment_runtime.py")
        or not proof["full3340_rank_identity"]
    ):
        raise ValueError("Complete source-bound balanced CPU ranking control required")
    with patch(
        "casmi_ml.portable_streaming_pipeline.extend_streamed", new=extend_balanced
    ):
        return original_run(recipe, data_dir, coconut, output, force_partitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--force-partitions", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
