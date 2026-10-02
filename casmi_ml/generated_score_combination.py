"""Predeclared combination of measured length, frequency and frozen critic scores."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.generated_calibration import calibrated_order
from casmi_ml.generation_frequency_ranking import ranked_candidates
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ROOT, freeze

SOURCE = Path("artifacts/research_loop/rounds/0005_coverage")
GENERATED = (
    ROOT
    / "generation/researchdev_samples128_limitall_stable_v2_98194888a437_frequency_v1.json"
)
SCORES = Path("artifacts/research_loop/rounds/0042_generated_critic/scores.json")
VARIANTS = {
    "baseline": (0.0, 0.0, 0.0),
    "tokens_frequency": (1.0, 1.0, 0.0),
    "tokens_critic": (1.0, 0.0, 0.5),
    "frequency_critic": (0.0, 1.0, 0.5),
    "tokens_frequency_critic": (1.0, 0.5, 0.5),
    "tokens_frequency1_critic": (1.0, 1.0, 0.5),
}


def combined_order(candidates, scores, spec):
    alpha, frequency_weight, critic_weight = spec
    ordered = calibrated_order(candidates, alpha, "sampled_tokens")
    lookup = {c["key"]: c for c in candidates}
    if frequency_weight:
        ordered = ranked_candidates([lookup[k] for k in ordered], frequency_weight)
    return rerank(
        ordered,
        {},
        [],
        critic_weight,
        top_n=max(1, len(ordered)),
        fragment_scores=scores,
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "source_directory": str(SOURCE),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "variants": VARIANTS,
            "composition_order": "Measured token length then frequency then fingerprint critic",
            "critic": "Frozen training-only fingerprint critic from0042; graph excluded by prior negative evidence",
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "prefix": 3,
            "slots": 5,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "independent_acceptance": False,
            "cohort": "repeated_development",
            "new_gpu_training": False,
        },
    )
    generated = json.loads(GENERATED.read_text())
    by_key = {r["key"]: r["candidates"] for r in generated}
    if len(generated) != 2000 or len(by_key) != 2000:
        raise ValueError("Full2000 generated cache required")
    scores = json.loads(SCORES.read_text())
    ordered = {
        name: {
            key: combined_order(
                candidates, scores.get(key, {}).get("fingerprint", {}), spec
            )
            for key, candidates in by_key.items()
        }
        for name, spec in VARIANTS.items()
    }
    report = {}
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
        report[mode] = {}
        for name in VARIANTS:
            ranks, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                ranks[key] = (
                    insert_generated(selected["ranking"], ordered[name][key], 3, 5)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (ordered[name][key] if allowed else [])
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
