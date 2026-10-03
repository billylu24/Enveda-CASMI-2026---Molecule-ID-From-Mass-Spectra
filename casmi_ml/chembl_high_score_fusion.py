"""Keep corrected preselection priority when ranking an existing high-confidence tail."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_critic_slots import score_cache_key
from casmi_ml.chembl_routed_combination import HIGH
from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import rrf
from casmi_ml.research_protocol import ROOT, freeze


def fused_tail(prior, current, native, values):
    """Equal fixed RRF; preserve existing tail if the new leader lacks its margin."""
    critic = sorted(native, key=lambda key: (-values[key], key))
    order = rrf([critic, native], [0.5, 0.5])
    if order and values[order[0]] > values[prior[0]] + 0.05:
        return insert_generated(prior, order, 10, 3), True
    return current, False


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((HIGH / "protocol.json").read_text())
    proposals_path = Path(config["proposal_path"])
    if digest(proposals_path) != config["native_proposals_sha256"]:
        raise ValueError("Frozen proposals changed")
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "incumbent_rankings_sha256": digest(incumbent / "selected_rankings.json"),
            "original_rankings_sha256": digest(incumbent / "low_rankings.json"),
            "high_protocol_sha256": digest(HIGH / "protocol.json"),
            "high_scores_sha256": digest(HIGH / "critic_scores.json"),
            "proposals_sha256": digest(proposals_path),
            "encoder_sha256": config["encoder_sha256"],
            "critic_sha256": config["critic_sha256"],
            "rule": "Only already inserted0149 high tail; same corrected preselect100, equal0.5 RRF critic and corrected ordinal ranking. Fused actual first must still exceed original first critic+.05; otherwise keep0149 tail. Preserve low arm and first10; slots3. Considered candidate pools identical to0149.",
            "cohort": "repeated_development",
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "independent_acceptance": False,
            "new_training": False,
            "new_sampling": False,
        },
    )
    baseline = json.loads((incumbent / "selected_rankings.json").read_text())
    original = json.loads((incumbent / "low_rankings.json").read_text())["baseline"]
    proposals = json.loads(proposals_path.read_text())
    pairs = json.loads((HIGH / "critic_scores.json").read_text())
    binding = {key: config[key] for key in ("encoder_sha256", "critic_sha256")}
    report, selected, statistics = {}, {}, {}
    for mode in ("unknown", "known"):
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        base = pd.read_csv(incumbent / f"{mode}_low_fragment_high_tail.csv").set_index(
            "key"
        )
        if set(base.index) != set(baseline[mode]):
            raise ValueError("Baseline paired keys differ")
        selected[mode] = {}
        statistics[mode] = {
            "eligible_groups": 0,
            "fused_margin_pass": 0,
            "changed_groups": 0,
        }
        for key, current in baseline[mode].items():
            prior = original[mode][key]
            result = current
            if confidence[key] >= 0.5 and current != prior:
                statistics[mode]["eligible_groups"] += 1
                native = [c for c in proposals[key] if c not in set(prior)][:100]
                values = pairs[score_cache_key(key, prior[0], native, binding)]
                result, passed = fused_tail(prior, current, native, values)
                statistics[mode]["fused_margin_pass"] += passed
            if result[:10] != current[:10] or (
                confidence[key] < 0.5 and result != current
            ):
                raise ValueError("Protected ranks changed")
            selected[mode][key] = result
            statistics[mode]["changed_groups"] += result != current
        report[mode] = {}
        for name, orders in (
            ("baseline", baseline[mode]),
            ("high_score_fusion", selected[mode]),
        ):
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
                raise ValueError("Frozen0149 full rank and CSV differ")
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
