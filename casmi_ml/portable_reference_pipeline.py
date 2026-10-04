"""Frozen scientific236 with exact source-bound train-only reference preprocessing."""

import argparse
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

from casmi_ml.metfrag import digest
from casmi_ml.portable_streaming_pipeline import run as streaming_run
from casmi_ml.train_reference_cache import build_cached_reference


def run(recipe, data_dir, coconut, output):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "train_reference_runtime"
    ]
    if (
        proof["source_sha256"] != digest("casmi_ml/train_reference_cache.py")
        or not proof["full400_rows_sparse_entries_and_members_exact"]
    ):
        raise ValueError("Complete source-bound train-reference cache proof required")
    config = json.loads(recipe.read_text())
    if config.get("reference_runtime") != {"scan": "shared_union_v1"}:
        raise ValueError("Original shared union reference rule required")
    build = partial(
        build_cached_reference, store=recipe.parent / "train_reference_cache"
    )
    with patch("casmi_ml.reference_reuse.build_shared_reference", new=build):
        return streaming_run(recipe, data_dir, coconut, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
