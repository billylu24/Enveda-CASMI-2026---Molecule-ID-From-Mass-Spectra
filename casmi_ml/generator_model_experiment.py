"""Fixed sampling and routing comparison of a decoder checkpoint against incumbent."""

import argparse
import fcntl
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.generation_experiment import generate
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ROOT, freeze


def run(output, checkpoint, limit=None):
    output, checkpoint = Path(output), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    source = Path("artifacts/research_loop/rounds/0005_coverage")
    baseline_path = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_directory": str(source),
            "source_report_sha256": digest(source / "report.json"),
            "baseline_generated_sha256": digest(baseline_path),
            "generator_checkpoint": str(checkpoint),
            "generator_sha256": digest(checkpoint),
            "samples": 128,
            "limit": limit,
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "prefix": 3,
            "slots": 5,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
        },
    )
    with Path("artifacts/research_loop/gpu.lock").open("a") as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generated = generate(
            ROOT, checkpoint=checkpoint, samples=128, stable_sampling=True, limit=limit
        )
    if limit is None and len(generated) != 2000:
        raise ValueError("Full structural cohort must contain 2000 molecules")
    by_key = {r["key"]: [c["key"] for c in r["candidates"]] for r in generated}
    original = {r["key"]: r for r in json.loads(baseline_path.read_text())}
    if limit is not None:

        def summarize(rows):
            stats = {
                name: sum(r["statistics"][name] for r in rows)
                for name in [
                    "samples",
                    "valid",
                    "mass_matching",
                    "unique_mass_matching",
                ]
            }
            stats.update(
                queries_with_candidates=sum(bool(r["candidates"]) for r in rows),
                exact_structure_hits=sum(
                    any(c["key"] == r["key"] for c in r["candidates"]) for r in rows
                ),
            )
            return stats

        result = {
            "molecules": len(generated),
            "baseline": summarize([original[r["key"]] for r in generated]),
            "selected": summarize(generated),
            "diagnostic_only": True,
            "independent_acceptance": False,
            "sampling": "128 frozen trajectories per query; same spectrum seeds and conditions",
        }
        write_json(output / "report.json", result)
        return result
    report = {}
    for mode in ["unknown", "known"]:
        records = json.loads((source / f"{mode}_records.json").read_text())
        if {r["key"] for r in records} != set(by_key):
            raise ValueError("Generator and retrieval molecule keys differ")
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        observed = set(
            pd.read_parquet(
                Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                / mode
                / "rows.parquet",
                columns=["inchikey14"],
            ).inchikey14
        )
        report[mode] = {}
        for variant in ["baseline", "model"]:
            ranks, pools = {}, {}
            for row in records:
                if mode == "known" and not row["known"]:
                    continue
                key = row["key"]
                base = row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                candidates = (
                    [c["key"] for c in original[key]["candidates"]]
                    if variant == "baseline"
                    else by_key[key]
                )
                ranks[key] = (
                    insert_generated(selected["ranking"], candidates, 3, 5)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (candidates if allowed else [])
            m, per = metrics(ranks, pools)
            report[mode][variant] = m
            per.to_csv(output / f"{mode}_{variant}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.checkpoint, a.limit), indent=2))


if __name__ == "__main__":
    main()
