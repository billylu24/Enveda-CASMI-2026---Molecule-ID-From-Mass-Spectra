"""Frozen same-science236 overlap entry with complete source-bound controls."""

import argparse
import json
from pathlib import Path

import torch

from casmi_ml.metfrag import digest
from casmi_ml.portable_overlap_pipeline import run as overlap_run

SOURCES = (
    "generation_handoff_stream",
    "generation_fragment_overlap",
    "portable_overlap_pipeline",
)


def run(recipe, data_dir, coconut, output, force_partitions=False):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "overlap_runtime"
    ]
    if (
        proof["full_rank_matches"] != {"unknown": 2000, "known": 1340}
        or not proof["all400_original_generation_handoffs_and_events_exact"]
    ):
        raise ValueError("Complete original3340 scoring and400 event controls required")
    for name in SOURCES:
        if proof["sources"][name] != digest("casmi_ml/" + name + ".py"):
            raise ValueError("Frozen overlap source changed: " + name)
    if proof["original_generation_source_sha256"] != digest(
        "casmi_ml/generation_inference.py"
    ):
        raise ValueError("Original per-molecule generator source changed")
    if torch.cuda.device_count() < 2 and not force_partitions:
        from casmi_ml.portable_balanced_cpu_pipeline import run as balanced_run

        return balanced_run(recipe, data_dir, coconut, output)
    return overlap_run(recipe, data_dir, coconut, output, force_partitions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--force-partitions", action="store_true")
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
