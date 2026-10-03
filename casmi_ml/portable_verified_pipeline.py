"""Portable frozen scientific inference with verified pure validation memoization."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from casmi_ml.cached_submission_validation import validate_submission
from casmi_ml.metfrag import digest
from casmi_ml.research_pipeline import predict


def run(recipe, data_dir, coconut, output):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "validation_runtime"
    ]
    if (
        digest("casmi_ml/cached_submission_validation.py")
        != proof["cache_source_sha256"]
        or digest("casmi_ml/inference.py") != proof["original_validator_source_sha256"]
    ):
        raise ValueError("Frozen verified validator source changed")
    with ExitStack() as stack:
        for module in (
            "chembl_routed_inference",
            "secondary_inference",
            "generation_inference",
        ):
            stack.enter_context(
                patch(
                    "casmi_ml." + module + ".validate_submission",
                    new=validate_submission,
                )
            )
        return predict(recipe, data_dir, coconut, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
