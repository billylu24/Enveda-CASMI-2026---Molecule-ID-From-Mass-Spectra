"""Source-bound unchanged236 entry with verified one-thread generation workers."""

import argparse
import json
from pathlib import Path

from casmi_ml.metfrag import digest
from casmi_ml.portable_overlap_cpu_pipeline import run as cpu_run

SOURCES = (
    "generation_handoff_cpu_worker",
    "portable_overlap_cpu_pipeline",
    "portable_verified_overlap_cpu_pipeline",
)


def run(recipe, data_dir, coconut, output, force_partitions=False):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())[
        "generation_cpu_runtime"
    ]
    if not (
        proof["full_rank_matches"] == {"unknown": 2000, "known": 1340}
        and proof["all400_generation_events_exact"]
        and proof["all400_generation_csv_audit_fullhandoffs_exact"]
        and proof["all400_top25_exact"]
        and proof["all265_raw_physical_dictionaries_exact"]
        and proof["torch_cpu_threads"] == 1
    ):
        raise ValueError("Complete unchanged236 CPU runtime controls required")
    for name in SOURCES:
        if proof["sources"][name] != digest("casmi_ml/" + name + ".py"):
            raise ValueError("Frozen CPU runtime source changed: " + name)
    return cpu_run(recipe, data_dir, coconut, output, force_partitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--force-partitions", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
