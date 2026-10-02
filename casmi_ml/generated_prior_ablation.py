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
from casmi_ml.generated_second_reference import select_prefix
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ROOT, freeze


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "baseline": (0.25, 0.25),
        "no_formula": (0.25, 0),
        "no_chemistry": (0, 0.25),
        "no_weak_priors": (0, 0),
    }
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "incumbent_report_sha256": digest(incumbent / "report.json"),
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
            "rule": "Freeze0047 routing/slots/critic; remove formula or chemistry support ranking priors separately and together",
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
    orderings = {
        variant: {
            key: combined_order(
                candidates,
                scores.get(key, {}).get("fingerprint", {}),
                (1, 0, 0.5),
                *weights,
            )
            for key, candidates in by_key.items()
        }
        for variant, weights in variants.items()
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
            ordered = orderings[variant]
            slots = 5
            ranks, pools, moved = {}, {}, 0
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                prefix = select_prefix(
                    base["ranking"], observed, confidence[key], "second_unreferenced"
                )
                if not allowed:
                    prefix = 2
                insert = True
                moved += int(
                    insert
                    and prefix == 1
                    and any(k not in set(selected["ranking"]) for k in ordered[key])
                )
                ranks[key] = (
                    insert_generated(selected["ranking"], ordered[key], prefix, slots)
                    if insert
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (ordered[key] if insert else [])
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
    parser.add_argument("--incumbent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.incumbent), indent=2))


if __name__ == "__main__":
    main()
