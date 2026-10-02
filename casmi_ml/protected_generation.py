"""Counterfactual high-confidence slots while preserving a frozen retrieval prefix."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.research_protocol import ROOT, freeze


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mass = Path("artifacts/research_loop/rounds/0001_mass_v2")
    generated_path = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
    generated = {
        r["key"]: [c["key"] for c in r["candidates"]]
        for r in json.loads(generated_path.read_text())
    }
    variants = {
        "baseline": None,
        **{f"protected_{p}_{n}": (p, n) for p in [5, 10, 15, 20] for n in [1, 3, 5]},
    }
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "mass_report_sha256": digest(mass / "report.json"),
            "generation_sha256": digest(generated_path),
            "variants": variants,
            "holdout_used": False,
            "scope": "Open high-confidence generation only after prefix; existing low-confidence slots5/5 retained",
        },
    )
    report = {}
    diagnostics = {}
    for mode in ["unknown", "known"]:
        rows = json.loads((mass / f"{mode}_records.json").read_text())
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        selected = [r for r in rows if mode == "unknown" or r["known"]]
        diagnostics[mode] = {
            "protected": sum(confidence[r["key"]] >= 0.5 for r in selected),
            "pure_generated_truths_in_protected": sum(
                confidence[r["key"]] >= 0.5 and r["key"] in generated[r["key"]]
                for r in selected
            ),
        }
        report[mode] = {}
        for name, spec in variants.items():
            ranks, pools = {}, {}
            for row in selected:
                key = row["key"]
                base = row["variants"]["charge_aware_union"]
                if confidence[key] < 0.5:
                    rank = insert_generated(base["ranking"], generated[key], 5, 5)
                elif spec:
                    rank = insert_generated(base["ranking"], generated[key], *spec)
                else:
                    rank = base["ranking"]
                ranks[key] = rank
                pools[key] = base["pool"] + generated[key]
            m, per = metrics(ranks, pools)
            report[mode][name] = m
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "report.json", report)
    write_json(output / "diagnostics.json", diagnostics)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))


if __name__ == "__main__":
    main()
