"""Diagnostic-ion and neutral-loss ranking controls against adaptive generation."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from casmi_ml.chemistry import extract_evidence, rerank, rule_manifest, score_candidate
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

VARIANTS = {"baseline": ("combined", 0.0)}
VARIANTS.update(
    {
        f"{component}_{weight}": (component, weight)
        for component in ("diagnostic", "loss", "combined")
        for weight in (0.25, 0.5, 1.0)
    }
)
COLUMNS = [
    "inchikey14",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "precursor_mz",
    "adduct",
    "instrument_type",
    "ionization_mode",
]


def evidence_order(ordered, existing, scores, component, weight):
    novel = [key for key in ordered if key not in set(existing)]
    selected = {
        key: values[component] + (values["combination"] if component == "loss" else 0)
        if component != "combined"
        else values["score"]
        for key, values in scores.items()
    }
    return rerank(
        novel, {}, [], weight, top_n=max(1, len(novel)), fragment_scores=selected
    )


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "chemistry_sha256": digest(Path("casmi_ml/chemistry.py")),
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "incumbent": str(incumbent),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": VARIANTS,
            "rules": rule_manifest(),
            "calibration": {
                "token_length_exponent": 1,
                "fingerprint_critic_weight": 0.5,
            },
            "adaptive_prefix": "second_unreferenced",
            "slots": 5,
            "candidate_scope": "novel generated candidates only; retrieval order preserved",
            "open_protected": True,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "cohort": "repeated_development",
            "new_gpu_training": False,
        },
    )
    started = time.monotonic()
    generated = json.loads(GENERATED.read_text())
    candidates = {r["key"]: r["candidates"] for r in generated}
    if len(generated) != 2000 or len(candidates) != 2000:
        raise ValueError("Require fixed full 2000 molecules")
    critic = json.loads(SCORES.read_text())
    ordered = {
        k: combined_order(v, critic.get(k, {}).get("fingerprint", {}), (1, 0, 0.5))
        for k, v in candidates.items()
    }
    # Only the grouping key and observable spectral fields are read, never truth structure/formula.
    frame = pq.read_table(
        ROOT / "researchdev.parquet", columns=COLUMNS, use_threads=False
    ).to_pandas()
    groups = frame.groupby("inchikey14", sort=True).indices
    if set(groups) != set(candidates):
        raise ValueError("Spectral and candidate molecule keys differ")
    scores, matched = {}, {}
    for key, values in candidates.items():
        records = (
            frame.iloc[groups[key]].drop(columns=["inchikey14"]).to_dict("records")
        )
        evidences = [extract_evidence(r) for r in records]
        matched[key] = any(e["matches"] for e in evidences)
        scores[key] = {
            c["key"]: score_candidate(evidences, c["smiles"]) for c in values
        }
    write_json(output / "scores.json", scores)
    report = {}
    for mode in ("unknown", "known"):
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        if {r["key"] for r in rows} != set(candidates):
            raise ValueError("Retrieval and generated molecule keys differ")
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
        for name, (component, weight) in VARIANTS.items():
            ranks, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                prefix = select_prefix(
                    base["ranking"], observed, confidence[key], "second_unreferenced"
                )
                chemical = evidence_order(
                    ordered[key], selected["ranking"], scores[key], component, weight
                )
                ranks[key] = (
                    insert_generated(selected["ranking"], chemical, prefix, 5)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (ordered[key] if allowed else [])
            result, per = metrics(ranks, pools)
            if name == "baseline":
                expected = (
                    pd.read_csv(incumbent / f"{mode}_second_unreferenced.csv")
                    .set_index("key")
                    .sort_index()
                )
                actual = per.set_index("key").sort_index()
                # CSV round trips can change last-bit float representation.
                if (
                    set(actual.index) != set(expected.index)
                    or (
                        actual[["reciprocal_rank", "top1"]]
                        - expected[["reciprocal_rank", "top1"]]
                    )
                    .abs()
                    .max()
                    .max()
                    > 1e-12
                ):
                    raise ValueError(
                        "Baseline must reproduce current incumbent paired metrics"
                    )
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostics"] = {
        "molecules_with_matched_evidence": sum(matched.values()),
        "candidate_queries_with_different_combined_scores": sum(
            len({v["score"] for v in rows.values()}) > 1 for rows in scores.values()
        ),
        "elapsed_seconds": time.monotonic() - started,
        "independent_acceptance": False,
    }
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
