"""Protect a reference-supported second retrieval candidate before moving generated slots."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import (
    GENERATED,
    SCORES,
    SOURCE,
    combined_order,
)
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ROOT, freeze


def select_prefix(ranking, observed, confidence, variant):
    if variant == "baseline":
        return 2
    unsupported = len(ranking) >= 2 and ranking[1] not in observed
    if variant == "second_unreferenced":
        return 1 if unsupported else 2
    if variant == "second_unreferenced_lowconf":
        return 1 if unsupported and confidence < 0.5 else 2
    raise ValueError("Unknown second-reference strategy")


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variants = ["baseline", "second_unreferenced", "second_unreferenced_lowconf"]
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "generated_sha256": digest(GENERATED),
            "scores_sha256": digest(SCORES),
            "source_directory": str(SOURCE),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "variants": variants,
            "calibration": {
                "token_length_exponent": 1,
                "fingerprint_critic_weight": 0.5,
            },
            "reference_guard": {"topn": 1, "threshold": 0},
            "open_protected": True,
            "slots": 5,
            "rule": "Move generated candidates after top1 only when original second retrieval structure has no retained reference spectrum; otherwise preserve top2",
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "cohort": "repeated_development",
        },
    )
    generated = json.loads(GENERATED.read_text())
    by_key = {r["key"]: r["candidates"] for r in generated}
    if len(generated) != 2000 or len(by_key) != 2000:
        raise ValueError("Full2000 required")
    scores = json.loads(SCORES.read_text())
    ordered = {
        k: combined_order(v, scores.get(k, {}).get("fingerprint", {}), (1, 0, 0.5))
        for k, v in by_key.items()
    }
    report, diagnostics = {}, {}
    for mode in ["unknown", "known"]:
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        if {r["key"] for r in rows} != set(by_key):
            raise ValueError("Cohort keys differ")
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
            ranks, pools, moved = {}, {}, 0
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                prefix = select_prefix(
                    base["ranking"], observed, confidence[key], variant
                )
                moved += int(
                    allowed
                    and prefix == 1
                    and any(k not in set(selected["ranking"]) for k in ordered[key])
                )
                ranks[key] = (
                    insert_generated(selected["ranking"], ordered[key], prefix, 5)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (ordered[key] if allowed else [])
            result, per = metrics(ranks, pools)
            report[mode][variant] = result
            diagnostics[mode][variant] = {
                "queries_with_earlier_novel_generation": moved
            }
            per.to_csv(output / f"{mode}_{variant}.csv", index=False)
    write_json(output / "diagnostics.json", diagnostics)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
