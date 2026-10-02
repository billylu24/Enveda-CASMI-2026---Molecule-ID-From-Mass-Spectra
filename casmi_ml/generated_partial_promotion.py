"""Fixed0062 gate with fewer novel generated candidates moved before retrieval."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.generated_first_critic import promotion_prefix
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


def partial_promotion(base, generated, prefix, front, slots=5):
    if not 1 <= front <= slots or prefix < 0:
        raise ValueError("Invalid partial promotion slots")
    novel = [key for key in dict.fromkeys(generated) if key not in set(base)][:slots]
    return novel[:front] + base[:prefix] + novel[front:] + base[prefix:]


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    pair_path = Path(
        "artifacts/research_loop/rounds/0061_generated_first_critic/scores.json"
    )
    variants = {"baseline": 5, "front1": 1, "front3": 3}
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "generated_sha256": digest(GENERATED),
            "scores_sha256": digest(SCORES),
            "promotion_scores_sha256": digest(pair_path),
            "variants": variants,
            "gate": {"confidence": 0.2, "margin": 0.05},
            "slots": 5,
            "rule": "Freeze0062; on promotion move only firstN novel candidates before retrieval; other novel candidates retain0047 prefix",
            "cohort": "repeated_development",
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "new_training": False,
        },
    )
    generated = json.loads(GENERATED.read_text())
    scores = json.loads(SCORES.read_text())
    ordered = {
        r["key"]: combined_order(
            r["candidates"],
            scores.get(r["key"], {}).get("fingerprint", {}),
            (1, 0, 0.5),
        )
        for r in generated
    }
    pairs = json.loads(pair_path.read_text())
    if len(generated) != 2000 or len(ordered) != 2000:
        raise ValueError("Full2000 required")
    report = {}
    for mode in ("unknown", "known"):
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        if {r["key"] for r in rows} != set(ordered):
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
        report[mode] = {}
        for name, front in variants.items():
            ranks, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                protected = protects_reference(
                    base["ranking"], observed, confidence[key]
                )
                selected = base if protected else row["variants"]["expansion_1"]
                prefix = (
                    select_prefix(
                        base["ranking"],
                        observed,
                        confidence[key],
                        "second_unreferenced",
                    )
                    if protected
                    else 2
                )
                ranking, candidates = selected["ranking"], ordered[key]
                effective = prefix
                novel = [k for k in candidates if k not in set(ranking)]
                if confidence[key] < 0.2 and ranking and novel:
                    effective = promotion_prefix(
                        ranking,
                        candidates,
                        pairs[key + ":" + ranking[0] + ":" + novel[0]],
                        prefix,
                        0.05,
                    )
                ranks[key] = (
                    partial_promotion(ranking, candidates, prefix, front)
                    if effective == 0
                    else insert_generated(ranking, candidates, prefix, 5)
                )
                pools[key] = selected["pool"] + candidates
            result, per = metrics(ranks, pools)
            if name == "baseline":
                expected = (
                    pd.read_csv(incumbent / f"{mode}_confidence0.2_margin0.05.csv")
                    .set_index("key")
                    .sort_index()
                )
                actual = per.set_index("key").sort_index()
                if (
                    not actual.index.equals(expected.index)
                    or (
                        actual[["reciprocal_rank", "top1"]]
                        - expected[["reciprocal_rank", "top1"]]
                    )
                    .abs()
                    .max()
                    .max()
                    > 1e-12
                ):
                    raise ValueError("Paired baseline must match0062")
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.incumbent), indent=2))


if __name__ == "__main__":
    main()
