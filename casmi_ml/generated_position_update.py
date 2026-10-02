"""Matched insertion positions for selected decoder and frozen calibrated ranking."""

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

VARIANTS = {
    "baseline": (3, 5, False),
    "prefix2_slots3": (2, 3, False),
    "prefix2_slots5": (2, 5, False),
    "prefix3_slots8": (3, 8, False),
    "prefix3_slots10": (3, 10, False),
    "calibrated_prefix2_slots5": (2, 5, True),
    "calibrated_prefix3_slots8": (3, 8, True),
}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "generated_sha256": digest(GENERATED),
            "scores_sha256": digest(SCORES),
            "source_directory": str(SOURCE),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "variants": VARIANTS,
            "calibrated_ranking": {
                "token_length_exponent": 1,
                "frequency_weight": 0,
                "fingerprint_critic_weight": 0.5,
            },
            "reference_guard": {"topn": 1, "threshold": 0},
            "open_protected": True,
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
    baseline = {k: [c["key"] for c in v] for k, v in by_key.items()}
    calibrated = {
        k: combined_order(v, scores.get(k, {}).get("fingerprint", {}), (1, 0, 0.5))
        for k, v in by_key.items()
    }
    report = {}
    for mode in ["unknown", "known"]:
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        if {r["key"] for r in rows} != set(by_key):
            raise ValueError("Molecule keys differ")
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
        for name, (prefix, slots, use_calibrated) in VARIANTS.items():
            ranks, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                ordered = (calibrated if use_calibrated else baseline)[key]
                ranks[key] = (
                    insert_generated(selected["ranking"], ordered, prefix, slots)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (ordered if allowed else [])
            result, per = metrics(ranks, pools)
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
