"""Move the three frozen163 fragment-supported high slots after the protected first."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_critic_slots import score_cache_key
from casmi_ml.chembl_high_fragment_pilot import HIGH, fragment_structures
from casmi_ml.chembl_routed_inference import select_high_fragment
from casmi_ml.chembl_tail_controls import validate_tail
from casmi_ml.chemistry import rerank
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, freeze

BASE = Path("artifacts/research_loop/rounds/0149_chembl_routed_combination")


def promoted_slots(prior, proposed):
    """Keep all three frozen supported slots and original first; only change position."""
    return insert_generated(prior, proposed[:3], 1, 3)


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((HIGH / "protocol.json").read_text())
    source_protocol = json.loads((incumbent / "protocol.json").read_text())
    assert source_protocol["prefix"] == 3 and source_protocol["limit"] == 2000
    proposals_path = Path(config["proposal_path"])
    if digest(proposals_path) != config["native_proposals_sha256"]:
        raise ValueError("Frozen proposal source differs")
    if digest(DERIVED) != config["catalog_sha256"]:
        raise ValueError("Frozen catalog differs")
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "incumbent_directory": str(incumbent),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "incumbent_rankings_sha256": digest(incumbent / "selected_rankings.json"),
            "fragment_cache_sha256": digest(incumbent / "fragment_scores.json"),
            "fragment_protocol_sha256": digest(incumbent / "protocol.json"),
            "proposals_sha256": digest(proposals_path),
            "critic_scores_sha256": digest(HIGH / "critic_scores.json"),
            "rule": "Freeze163 low/fallback/remove-tail branches. Only163 fragment-supported high insertions: move identical three inserted candidates from after available original first3 to after protected original first1. Same strict actual-first fragment and critic+.05 group gates. No new candidates or thresholds; all100 evidence/pools/scores unchanged.",
            "new_training": False,
            "new_sampling": False,
            "new_java": False,
            "resource_scope": "Cached full-score accuracy diagnostic, not a runtime improvement or cold deployment guarantee",
            "cohort": "repeated_development",
            "truth_used_only_in_metrics": True,
            "independent_acceptance": False,
            "holdout_used": False,
        },
    )
    baseline = json.loads((incumbent / "selected_rankings.json").read_text())
    routed = json.loads((BASE / "selected_rankings.json").read_text())
    original = json.loads((BASE / "low_rankings.json").read_text())["baseline"]
    proposals = json.loads(proposals_path.read_text())
    pairs = json.loads((HIGH / "critic_scores.json").read_text())
    cache = json.loads((incumbent / "fragment_scores.json").read_text())
    binding = {key: config[key] for key in ("encoder_sha256", "critic_sha256")}
    external = (
        pd.read_parquet(DERIVED, columns=["inchikey14", "normalized_smiles"])
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    pubchem = pd.read_parquet(
        "external/pubchemlite/structures.parquet",
        columns=["inchikey14", "normalized_smiles"],
    )
    for row in pubchem.itertuples():
        lookup.setdefault(row.inchikey14, row.normalized_smiles)
    del pubchem
    generated = {
        r["key"]: {c["key"]: c["smiles"] for c in r["candidates"]}
        for r in json.loads(GENERATED.read_text())
    }
    report, selected, statistics = {}, {}, {}
    for mode in ("unknown", "known"):
        if mode == "known":
            lookup.update(candidate_lookup(ROOT, "researchdev", "known"))
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        base = pd.read_csv(incumbent / f"{mode}_high_relative_fragment.csv").set_index(
            "key"
        )
        selected[mode] = {}
        stats = {
            "reconstructed_high_groups": 0,
            "supported_high_groups": 0,
            "moved_slots": 0,
            "changed_groups": 0,
        }
        for key, current in baseline[mode].items():
            prior = original[mode][key]
            result = current
            if confidence[key] >= 0.5 and routed[mode][key] != prior:
                native = [k for k in proposals[key] if k not in set(prior)][:100]
                values = pairs[score_cache_key(key, prior[0], native, binding)]
                candidates = sorted(native, key=lambda k: (-values[k], k))
                structures = fragment_structures(
                    key, prior[0], candidates, external, lookup, generated
                )
                cache_key = score_cache_key(
                    "high_relative_merged_actual_first_v2:" + key,
                    prior[0],
                    candidates,
                    {
                        "structures": structures,
                        "dependencies": source_protocol["dependencies_sha256"],
                    },
                )
                item = cache[cache_key]
                reconstructed, _, status = select_high_fragment(
                    prior,
                    routed[mode][key],
                    candidates,
                    values,
                    item["scores"],
                    item["budget_fallback"],
                )
                if reconstructed != current:
                    raise ValueError("Frozen163 high ranking reconstruction differs")
                stats["reconstructed_high_groups"] += 1
                if status == "high_fragment_inserted":
                    proposed = rerank(
                        candidates,
                        {},
                        [],
                        0.5,
                        top_n=len(candidates),
                        fragment_scores=item["scores"],
                    )
                    result = promoted_slots(prior, proposed)
                    stats["supported_high_groups"] += 1
                    stats["moved_slots"] += (
                        min(3, len(proposed)) if result != current else 0
                    )
            validate_tail(prior, current, result, confidence[key], prefix=1)
            selected[mode][key] = result
            stats["changed_groups"] += result != current
        statistics[mode] = stats
        report[mode] = {}
        for name, orders in [
            ("baseline", baseline[mode]),
            ("high_prefix1", selected[mode]),
        ]:
            per = base.copy()
            for key, order in orders.items():
                rank = order.index(key) + 1 if key in order else 0
                per.loc[key, ["reciprocal_rank", "top1", "top5", "top25"]] = [
                    1 / rank if 0 < rank <= 25 else 0,
                    int(rank == 1),
                    int(0 < rank <= 5),
                    int(0 < rank <= 25),
                ]
            if (
                name == "baseline"
                and abs(per.reciprocal_rank.mean() - base.reciprocal_rank.mean())
                > 1e-12
            ):
                raise ValueError("Frozen163 ranking metrics differ")
            report[mode][name] = {
                "molecules": len(per),
                "candidate_recall": float(per.covered.mean()),
                "mrr25": float(per.reciprocal_rank.mean()),
                **{k: float(per[k].mean()) for k in ("top1", "top5", "top25")},
                "conditional_mrr25": float(
                    per.loc[per.covered == 1, "reciprocal_rank"].mean()
                ),
            }
            per.reset_index().to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "selected_rankings.json", selected)
    report.update(
        diagnostic_only=False,
        independent_acceptance=False,
        deployment_required=True,
        diagnostics=statistics,
    )
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
