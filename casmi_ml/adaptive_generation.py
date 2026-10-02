"""Use retained reference availability to choose a generated candidate insertion point."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ROOT, freeze


def adaptive_prefix(ranking, observed, confidence, strategy):
    if strategy == "baseline":
        return 3
    if strategy == "unreferenced_first":
        return 1 if ranking and ranking[0] not in observed else 3
    if strategy == "low_confidence":
        return 1 if confidence < 0.5 else 3
    if strategy == "unreferenced_or_low":
        return 1 if confidence < 0.5 or (ranking and ranking[0] not in observed) else 3
    raise ValueError("Unknown adaptive prefix strategy")


def run(output, source=Path("artifacts/research_loop/rounds/0005_coverage")):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated_path = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
    generated = {
        r["key"]: [c["key"] for c in r["candidates"]]
        for r in json.loads(generated_path.read_text())
    }
    variants = [
        "baseline",
        "unreferenced_first",
        "low_confidence",
        "unreferenced_or_low",
    ]
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_directory": str(source),
            "source_report_sha256": digest(source / "report.json"),
            "generated_sha256": digest(generated_path),
            "variants": variants,
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "slots": 5,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "rule": "Change prefix3 to prefix1 only by original top1 retained-reference membership or original confidence; expansion branch still bars generation",
        },
    )
    report, diagnostics = {}, {}
    for mode in ["unknown", "known"]:
        rows = json.loads((source / f"{mode}_records.json").read_text())
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
        report[mode], diagnostics[mode] = {}, {}
        for variant in variants:
            ranks, pools = {}, {}
            moved = 0
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key = row["key"]
                base = row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                chosen = base if allowed else row["variants"]["expansion_1"]
                prefix = adaptive_prefix(
                    base["ranking"], observed, confidence[key], variant
                )
                moved += int(
                    allowed
                    and prefix == 1
                    and any(c not in set(base["ranking"]) for c in generated[key])
                )
                ranks[key] = (
                    insert_generated(chosen["ranking"], generated[key], prefix, 5)
                    if allowed
                    else chosen["ranking"]
                )
                pools[key] = chosen["pool"] + (generated[key] if allowed else [])
            m, per = metrics(ranks, pools)
            report[mode][variant] = m
            diagnostics[mode][variant] = {
                "queries_with_earlier_novel_generation": moved
            }
            per.to_csv(output / f"{mode}_{variant}.csv", index=False)
    write_json(output / "report.json", report)
    write_json(output / "diagnostics.json", diagnostics)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))


if __name__ == "__main__":
    main()
