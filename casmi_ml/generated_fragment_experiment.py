"""Label-free MetFrag reranking of novel decoder candidates, with frozen routing."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import MetFrag, digest, score_group
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ROOT, freeze

SOURCE = Path("artifacts/research_loop/rounds/0005_coverage")
JAVA = "external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java"
SPECTRAL_COLUMNS = [
    "inchikey14",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "precursor_mz",
    "adduct",
    "instrument_type",
    "ionization_mode",
]


def insert_fragment_ranked(base, candidates, scores, weight):
    # Score only insertable structures. Existing retrieval entries cannot compete
    # for the five generated slots or change the protected retrieval prefix.
    existing = set(base)
    novel = list(
        dict.fromkeys(c["key"] for c in candidates if c["key"] not in existing)
    )
    ordered = rerank(
        novel, {}, [], weight, top_n=max(1, len(novel)), fragment_scores=scores
    )
    return insert_generated(base, ordered, 3, 5)


def run(output, generated_path, limit=200, seconds=1200):
    output, generated_path = Path(output), Path(generated_path)
    output.mkdir(parents=True, exist_ok=True)
    if limit is not None and limit < 1:
        raise ValueError("Positive pilot limit required")
    all_generated = json.loads(generated_path.read_text())
    if len(all_generated) != 2000 or len({r["key"] for r in all_generated}) != 2000:
        raise ValueError("Require complete 2000-molecule frozen generation cache")
    generated = all_generated if limit is None else all_generated[:limit]
    candidates = {r["key"]: r["candidates"] for r in generated}
    variants = {
        "baseline": 0.0,
        "fragment_0.25": 0.25,
        "fragment_0.5": 0.5,
        "fragment_1": 1.0,
    }
    jar = Path("external/metfrag/MetFragCommandLine-2.6.11.jar")
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "source_directory": str(SOURCE),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "generated_path": str(generated_path),
            "generated_sha256": digest(generated_path),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "jar_sha256": digest(jar),
            "java_sha256": digest(JAVA),
            "limit": limit,
            "variants": variants,
            "fragment_seconds": seconds,
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "prefix": 3,
            "slots": 5,
            "candidate_scope": "union of novel insertable candidates across routing modes",
            "partial_group_fallback": True,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "independent_acceptance": False,
        },
    )
    modes, required = {}, {}
    for mode in ["unknown", "known"]:
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        if {r["key"] for r in rows} != {r["key"] for r in all_generated}:
            raise ValueError("Generated and retrieval cohort keys differ")
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
        modes[mode] = []
        for row in rows:
            key = row["key"]
            if key not in candidates or (mode == "known" and not row["known"]):
                continue
            base = row["variants"]["baseline"]
            allowed = protects_reference(base["ranking"], observed, confidence[key])
            selected = base if allowed else row["variants"]["expansion_1"]
            modes[mode].append((key, selected, allowed))
            if allowed:
                novel = {
                    c["key"]: c["smiles"]
                    for c in candidates[key]
                    if c["key"] not in set(selected["ranking"])
                }
                if len(novel) > 1:
                    required.setdefault(key, {}).update(novel)
    # Ground-truth SMILES, formula and fingerprint are never loaded for scoring.
    frame = pq.read_table(
        ROOT / "researchdev.parquet", columns=SPECTRAL_COLUMNS, use_threads=False
    ).to_pandas()
    groups = frame.groupby("inchikey14", sort=True).indices
    score_path = output / "fragment_scores.json"
    results = json.loads(score_path.read_text()) if score_path.exists() else {}
    budget = StageBudget(
        output,
        "fragments",
        "generated_candidates",
        seconds,
        limit=seconds,
        lock_path=output / "fragment_job.lock",
    )
    fragmenter = MetFrag(jar, ROOT / "generated_metfrag_cache", java=JAVA)
    start = time.monotonic()
    try:
        for i, (key, lookup) in enumerate(required.items(), 1):
            if key in results:
                continue
            records = (
                frame.iloc[groups[key]].drop(columns=["inchikey14"]).to_dict("records")
            )
            scores, exhausted = score_group(
                fragmenter, records, lookup, deadline=budget.started + budget.allowance
            )
            results[key] = {"scores": scores, "budget_fallback": exhausted}
            write_json(score_path, results)
            if i % 10 == 0 or exhausted:
                print(
                    "fragments",
                    i,
                    "/",
                    len(required),
                    "seconds",
                    time.monotonic() - start,
                    flush=True,
                )
    finally:
        budget.close()
    report = {}
    for mode, rows in modes.items():
        report[mode] = {}
        for name, weight in variants.items():
            ranks, pools = {}, {}
            for key, selected, allowed in rows:
                scores = results.get(key, {}).get("scores", {})
                ranks[key] = (
                    insert_fragment_ranked(
                        selected["ranking"], candidates[key], scores, weight
                    )
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (
                    [c["key"] for c in candidates[key]] if allowed else []
                )
            result, per = metrics(ranks, pools)
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostic_only"] = limit is not None
    report["diagnostics"] = {
        "molecules": len(generated),
        "diagnostic_only": limit is not None,
        "independent_acceptance": False,
        "repeated_development": True,
        "queries_requiring_scoring": len(required),
        "queries_with_fragment_scores": sum(
            bool(r["scores"]) for r in results.values()
        ),
        "queries_with_differentiating_scores": sum(
            len(set(r["scores"].values())) > 1 for r in results.values()
        ),
        "budget_fallbacks": sum(r["budget_fallback"] for r in results.values()),
        "fragment_elapsed_seconds": time.monotonic() - start,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seconds", type=float, default=1200)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                args.generated,
                None if args.full else args.limit,
                args.seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
