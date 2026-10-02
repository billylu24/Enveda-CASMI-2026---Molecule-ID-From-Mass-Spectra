"""Development-only length calibration of decoder candidate log probabilities."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ROOT, freeze

SOURCE = Path("artifacts/research_loop/rounds/0005_coverage")


def calibrated_order(
    candidates,
    alpha,
    length_measure="canonical_smiles",
    chemical_weight=0.25,
    formula_weight=0.25,
):
    if not 0 <= chemical_weight <= 1 or not 0 <= formula_weight <= 1:
        raise ValueError("Chemical and formula weights must be between0 and1")
    if length_measure not in ["canonical_smiles", "sampled_tokens"]:
        raise ValueError("Unknown sequence length measure")
    if not 0 <= alpha <= 1:
        raise ValueError("Length exponent must be between zero and one")
    if alpha == 0:
        return [c["key"] for c in candidates]
    lookup = {c["key"]: c for c in candidates}
    if length_measure == "sampled_tokens" and any(
        c.get("best_sequence_tokens", 0) < 1 for c in candidates
    ):
        raise ValueError("Measured best-sequence token count required")
    order = sorted(
        lookup,
        key=lambda k: (
            -lookup[k]["log_probability"]
            / (
                lookup[k]["best_sequence_tokens"]
                if length_measure == "sampled_tokens"
                else max(1, len(lookup[k]["smiles"]))
            )
            ** alpha,
            k,
        ),
    )
    for field, weight in [
        ("chemical_score", chemical_weight),
        ("formula_support", formula_weight),
    ]:
        order = rerank(
            order,
            {},
            [],
            weight,
            top_n=max(1, len(order)),
            fragment_scores={k: c[field] for k, c in lookup.items()},
        )
    return order


def run(output, generated_path, length_measure="canonical_smiles"):
    output, generated_path = Path(output), Path(generated_path)
    output.mkdir(parents=True, exist_ok=True)
    generated = json.loads(generated_path.read_text())
    by_key = {r["key"]: r["candidates"] for r in generated}
    if len(generated) != 2000 or len(by_key) != 2000:
        raise ValueError("Full 2000-molecule generated cache required")
    prefix = "tokens" if length_measure == "sampled_tokens" else "length"
    variants = {"baseline": 0.0, f"{prefix}_0.5": 0.5, f"{prefix}_1": 1.0}
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "source_directory": str(SOURCE),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "generated_path": str(generated_path),
            "generated_sha256": digest(generated_path),
            "variants": variants,
            "length_measure": (
                "Measured token count of highest-probability sampled sequence, includingEOS excludingBOS"
                if length_measure == "sampled_tokens"
                else "canonical SMILES character count proxy; sampled token lengths unavailable"
            ),
            "chemical_weight": 0.25,
            "formula_weight": 0.25,
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "prefix": 3,
            "slots": 5,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "independent_acceptance": False,
        },
    )
    ordered = {
        name: {
            key: calibrated_order(candidates, alpha, length_measure)
            for key, candidates in by_key.items()
        }
        for name, alpha in variants.items()
    }
    report = {}
    for mode in ["unknown", "known"]:
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        if {r["key"] for r in rows} != set(by_key):
            raise ValueError("Generated and retrieval cohort keys differ")
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
        for name in variants:
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
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument(
        "--length-measure",
        choices=["canonical_smiles", "sampled_tokens"],
        default="canonical_smiles",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.generated, args.length_measure), indent=2))


if __name__ == "__main__":
    main()
