"""Protect retrieval candidates with actual reference spectra, never query truth."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.research_protocol import ROOT, freeze


def protects_reference(ranking, observed, confidence, topn=1, threshold=0.0):
    return confidence >= 0.5 or (
        confidence >= threshold and bool(set(ranking[:topn]) & observed)
    )


def run(source, output, open_protected=False, slot_ablation=False):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated_path = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
    generated = {
        r["key"]: [c["key"] for c in r["candidates"]]
        for r in json.loads(generated_path.read_text())
    }
    modes = {
        f"reference_{n}_{threshold:g}": (n, threshold)
        for n in [1, 3, 5]
        for threshold in [0.0, 0.05, 0.1, 0.2]
    }
    if open_protected:
        modes = {"reference_1_0": (1, 0.0)}
    if slot_ablation:
        if not open_protected:
            raise ValueError("Slot ablation requires open protected generation")
        modes = {
            f"slots_{prefix}_{slots}": (1, 0.0)
            for prefix, slots in [(1, 3), (1, 5), (3, 3), (3, 5), (5, 8), (5, 10)]
        }
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_directory": str(source),
            "source_report_sha256": digest(source / "report.json"),
            "generated_sha256": digest(generated_path),
            "modes": modes,
            "incumbent": "union+fragment+slots5/5",
            "open_protected": open_protected,
            "slot_ablation": slot_ablation,
            "rule": "If any original first N candidate has retained reference spectra and confidence >= threshold, use incumbent; otherwise expanded chemical ranking",
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
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
        observed = set(
            pd.read_parquet(
                Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                / mode
                / "rows.parquet",
                columns=["inchikey14"],
            ).inchikey14
        )
        variants = {"baseline": None, **modes}
        report[mode] = {}
        for name, spec in variants.items():
            prefix, slots = (
                map(int, name.split("_")[1:]) if name.startswith("slots_") else (5, 5)
            )
            rankings, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key = row["key"]
                base = row["variants"]["baseline"]
                incumbent = (
                    base["ranking"]
                    if confidence[key] >= 0.5 and not open_protected
                    else insert_generated(
                        base["ranking"], generated[key], prefix, slots
                    )
                )
                use_original = (
                    (spec is None and not slot_ablation)
                    or confidence[key] >= 0.5
                    or protects_reference(
                        base["ranking"], observed, confidence[key], *(spec or (1, 0.0))
                    )
                )
                rankings[key] = (
                    incumbent
                    if use_original
                    else row["variants"]["expansion_1"]["ranking"]
                )
                pools[key] = (
                    base["pool"] + generated[key]
                    if use_original
                    else row["variants"]["expansion_1"]["pool"]
                )
            m, per = metrics(rankings, pools)
            report[mode][name] = m
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--open-protected", action="store_true")
    p.add_argument("--slot-ablation", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(run(a.source, a.output, a.open_protected, a.slot_ablation), indent=2)
    )


if __name__ == "__main__":
    main()
