"""Development-only reliability gates on the frozen expanded candidate rankings."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.research_protocol import ROOT, freeze


def run(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    weights = [0.25, 0.5, 1.0]
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_directory": str(source),
            "source_report_sha256": digest(source / "report.json"),
            "thresholds": thresholds,
            "weights": weights,
            "protection": "confidence >= expansion threshold preserves incumbent union+fragment rank",
            "cohort_sha256": digest(ROOT / "researchdev.parquet"),
            "holdout_used": False,
            "uses_only_frozen_ranks": True,
        },
    )
    report = {}
    for mode in ["unknown", "known"]:
        rows = json.loads((source / f"{mode}_records.json").read_text())
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        report[mode] = {}
        variants = {
            "baseline": None,
            **{
                f"guard_{threshold:g}_{weight:g}": (threshold, weight)
                for threshold in thresholds
                for weight in weights
            },
        }
        output_rows = []
        for row in rows:
            values = {"baseline": row["variants"]["baseline"]}
            for name, spec in variants.items():
                if spec is None:
                    continue
                threshold, weight = spec
                values[name] = (
                    row["variants"]["baseline"]
                    if confidence[row["key"]] >= threshold
                    else row["variants"][f"expansion_{weight:g}"]
                )
            output_rows.append(
                {"key": row["key"], "known": row["known"], "variants": values}
            )
        for name in variants:
            selected = [r for r in output_rows if mode == "unknown" or r["known"]]
            rankings = {r["key"]: r["variants"][name]["ranking"] for r in selected}
            pools = {r["key"]: r["variants"][name]["pool"] for r in selected}
            result, per = metrics(rankings, pools)
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
