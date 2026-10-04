"""Source-bound scientific236 entry with exact peak-loop-local mass reuse."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.mass_reuse_fragment_runtime import bound_mass_reuse
from casmi_ml.metfrag import digest
from casmi_ml.portable_verified_overlap_cpu_pipeline import run as original_run


def run(recipe, data_dir, coconut, output, force_partitions=False):
    root = Path(recipe).parent
    proof = json.loads((root / "release_evidence.json").read_text())[
        "mass_reuse_runtime"
    ]
    if not (
        proof["full_rank_matches"] == {"unknown": 2000, "known": 1340}
        and proof["all400_generation_csv_audit_fullhandoffs_exact"]
        and proof["all400_top25_exact"]
        and proof["all265_raw_physical_dictionaries_exact"]
        and proof["boundary_match_comparisons_exact"] == 113400
    ):
        raise ValueError("Complete mass-reuse identity controls required")
    for name, expected in proof["sources"].items():
        if digest(root / "casmi_ml" / (name + ".py")) != expected:
            raise ValueError("Mass-reuse source changed: " + name)
    with bound_mass_reuse(root / "mass_reuse_classes", proof["classes"]):
        result = original_run(recipe, data_dir, coconut, output, force_partitions)
    path = Path(str(output) + ".report.json")
    report = json.loads(path.read_text())
    report["mass_reuse_runtime"] = {
        "classes": proof["classes"],
        "sources": proof["sources"],
        "original_deadlines_retained": True,
        "mass_reuse_scope": "same ordered mass sum once per unchanged fragment peak loop; no persistent fragment cache",
    }
    write_json(path, report)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--force-partitions", action="store_true")
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
