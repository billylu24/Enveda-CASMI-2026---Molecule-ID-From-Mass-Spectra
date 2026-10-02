"""Paired generated-structure frequency ranking with fixed decoder trajectories."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.generation_experiment import generate
from casmi_ml.generation_frequency_ranking import ranked_candidates
from casmi_ml.generation_slots import insert_generated
from casmi_ml.generation_temperature_experiment import BASELINE, CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ROOT, freeze

SOURCE = Path("artifacts/research_loop/rounds/0005_coverage")


def run(output, limit=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "baseline": 0.0,
        "frequency_0.25": 0.25,
        "frequency_0.5": 0.5,
        "frequency_1": 1.0,
    }
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "sampling_source_sha256": digest(Path(generate.__code__.co_filename)),
            "ranking_source_sha256": digest(
                Path(__file__).with_name("generation_frequency_ranking.py")
            ),
            "source_directory": str(SOURCE),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "generator_checkpoint": str(CHECKPOINT),
            "generator_sha256": digest(CHECKPOINT),
            "baseline_generated_sha256": digest(BASELINE),
            "generated_path": str(
                ROOT
                / "generation"
                / f"researchdev_samples128_limit{limit or 'all'}_stable_v2_{digest(CHECKPOINT)[:12]}_frequency_v1.json"
            ),
            "variants": variants,
            "samples_per_query": 128,
            "temperature": 0.8,
            "limit": limit,
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "candidate_statistics": "Measured count of finished valid mass-matching trajectories yielding same14-character structural key",
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "prefix": 3,
            "slots": 5,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
        },
    )
    budget = StageBudget(output, "generation", "frequency_sampling", 86400)
    try:
        rows = generate(
            ROOT,
            checkpoint=CHECKPOINT,
            samples=128,
            stable_sampling=True,
            limit=limit,
            deadline=budget.started + budget.allowance,
            track_frequency=True,
        )
    finally:
        budget.close()
    candidates = {r["key"]: r["candidates"] for r in rows}
    incumbent = {r["key"]: r for r in json.loads(BASELINE.read_text())}
    # New measurements must leave all control trajectories, structure keys and
    # likelihoods identical to incumbent; any mismatch invalidates paired evidence.
    matches = 0
    for row in rows:
        old = incumbent[row["key"]]
        control = [
            {
                k: v
                for k, v in c.items()
                if k not in ["sample_count", "best_sequence_tokens"]
            }
            for c in row["candidates"]
        ]
        if control != old["candidates"] or row["statistics"] != old["statistics"]:
            raise ValueError(
                "Frequency sampling changed the frozen baseline candidates"
            )
        if (
            sum(c["sample_count"] for c in row["candidates"])
            != row["statistics"]["mass_matching"]
        ):
            raise ValueError(
                "Frequency count does not account for every matching trajectory"
            )
        matches += 1
    if limit is None and len(rows) != 2000:
        raise ValueError("Full2000 required")
    ordered = {
        name: {
            key: ranked_candidates(values, weight) for key, values in candidates.items()
        }
        for name, weight in variants.items()
    }
    report = {}
    for mode in ["unknown", "known"]:
        records = json.loads((SOURCE / f"{mode}_records.json").read_text())
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
        for name in variants:
            ranks, pools = {}, {}
            for row in records:
                key = row["key"]
                if key not in candidates or (mode == "known" and not row["known"]):
                    continue
                base = row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                ranks[key] = (
                    insert_generated(selected["ranking"], ordered[name][key], 3, 5)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (ordered[name][key] if allowed else [])
            m, per = metrics(ranks, pools)
            report[mode][name] = m
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostic_only"] = limit is not None
    report["diagnostics"] = {
        "molecules": len(rows),
        "baseline_candidate_and_statistics_matches": matches,
        "queries_with_duplicate_structural_samples": sum(
            any(c["sample_count"] > 1 for c in r["candidates"]) for r in rows
        ),
        "mass_matching_trajectories": sum(
            r["statistics"]["mass_matching"] for r in rows
        ),
        "unique_mass_matching_structures": sum(len(r["candidates"]) for r in rows),
        "cohort": "repeated_development",
        "independent_acceptance": False,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, None if args.full else args.limit), indent=2))


if __name__ == "__main__":
    main()
