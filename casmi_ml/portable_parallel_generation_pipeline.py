"""Same verified reference/science pipeline with independently seeded dualGPU generation."""

import argparse
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

from casmi_ml.metfrag import digest
from casmi_ml.parallel_generation_runtime import parallel_predict
from casmi_ml.portable_reference_pipeline import run as original_run


def run(recipe, data_dir, coconut, output, force_partitions=False):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "parallel_generation_runtime"
    ]
    if (
        proof["source_sha256"] != digest("casmi_ml/parallel_generation_runtime.py")
        or proof["original_generation_source_sha256"]
        != digest("casmi_ml/generation_inference.py")
        or proof["full400_generation_handoffs_exact"] is not True
    ):
        raise ValueError(
            "Current source-bound complete generation partition proof required"
        )
    with patch(
        "casmi_ml.research_pipeline.generate_predict",
        new=partial(parallel_predict, force_partition=force_partitions),
    ):
        return original_run(recipe, data_dir, coconut, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--force-partitions", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
