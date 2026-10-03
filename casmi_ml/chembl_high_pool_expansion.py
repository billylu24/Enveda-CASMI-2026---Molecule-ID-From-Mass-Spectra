"""Expand only the calibrated high-confidence shortlist; freeze the0149 low arm."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml import chembl_critic_slots
from casmi_ml.chembl_routed_combination import HIGH
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, freeze

ORIGINAL = Path("artifacts/research_loop/rounds/0062_generated_first_gate")


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((HIGH / "protocol.json").read_text())
    proposals = Path(old["proposal_path"])
    for path, expected in (
        (proposals, old["native_proposals_sha256"]),
        (ROOT / "researchdev.parquet", old["development_sha256"]),
        (ORIGINAL / "report.json", old["incumbent_report_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError("Frozen high-confidence arm input changed")
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "worker_source_sha256": digest(Path(chembl_critic_slots.__file__)),
            "development_sha256": old["development_sha256"],
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "incumbent_rankings_sha256": digest(incumbent / "selected_rankings.json"),
            "original_report_sha256": digest(ORIGINAL / "report.json"),
            "high_protocol_sha256": digest(HIGH / "protocol.json"),
            "proposal_sha256": digest(proposals),
            "proposal_limit": 500,
            "prefix": 10,
            "slots": 3,
            "critic_margin": 0.05,
            "rule": "Only increase high confidence>=.5 corrected preselection100 to500; original0062 high retrieval/generation/first10 unchanged, critic+.05 to actual first, original critic ordering and3 tail slots unchanged. Preserve0149 low arm exactly.",
            "new_training": False,
            "new_sampling": False,
            "truth_used_only_in_metrics": True,
            "cohort": "repeated_development",
            "holdout_used": False,
            "independent_acceptance": False,
            "score_seconds": 3600,
        },
    )
    worker = output / "high_arm"
    if not (worker / "report.json").exists():
        chembl_critic_slots.run(
            worker,
            ORIGINAL,
            proposal_limit=500,
            proposal_path=proposals,
            confidence_scope="high",
        )
    new_ranks = json.loads((worker / "rankings.json").read_text())
    baseline_ranks = json.loads((incumbent / "selected_rankings.json").read_text())
    selected_ranks, report, changes = {}, {}, {}
    for mode in ("unknown", "known"):
        conf = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        base = (
            pd.read_csv(incumbent / f"{mode}_low_fragment_high_tail.csv")
            .set_index("key")
            .sort_index()
        )
        new = (
            pd.read_csv(worker / f"{mode}_margin005_prefix10.csv")
            .set_index("key")
            .sort_index()
        )
        original = (
            pd.read_csv(worker / f"{mode}_baseline.csv").set_index("key").sort_index()
        )
        if not base.index.equals(new.index) or not original.index.equals(base.index):
            raise ValueError("Complete identical paired keys required")
        low = np.asarray([conf[k] < 0.5 for k in new.index])
        if not np.allclose(new.loc[low], original.loc[low], rtol=0, atol=1e-12):
            raise ValueError("New high arm modifies low confidence scope")
        chosen = base.copy()
        chosen.loc[~low] = new.loc[~low]
        ranks = {}
        for key in base.index:
            ranks[key] = (
                baseline_ranks[mode][key]
                if conf[key] < 0.5
                else new_ranks[mode]["margin005_prefix10"][key]
            )
            if ranks[key][:10] != baseline_ranks[mode][key][:10]:
                raise ValueError("Protected first10 differs")
            position = ranks[key].index(key) + 1 if key in ranks[key] else 0
            expected = 1 / position if 0 < position <= 25 else 0
            if abs(chosen.loc[key, "reciprocal_rank"] - expected) > 1e-12:
                raise ValueError("Actual ranking differs from paired CSV")
        selected_ranks[mode] = ranks
        changes[mode] = sum(ranks[k] != baseline_ranks[mode][k] for k in ranks)
        report[mode] = {}
        for name, frame in (("baseline", base), ("high_pool500", chosen)):
            report[mode][name] = {
                "molecules": len(frame),
                "candidate_recall": float(frame.covered.mean()),
                "mrr25": float(frame.reciprocal_rank.mean()),
                **{k: float(frame[k].mean()) for k in ("top1", "top5", "top25")},
                "conditional_mrr25": float(
                    frame.loc[frame.covered == 1, "reciprocal_rank"].mean()
                ),
            }
            frame.reset_index().to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "selected_rankings.json", selected_ranks)
    report.update(
        diagnostic_only=False,
        independent_acceptance=False,
        deployment_required=True,
        diagnostics={
            "high_arm": json.loads((worker / "report.json").read_text())["diagnostics"],
            "changed_groups": changes,
            "low_unchanged": True,
            "first10_unchanged": True,
        },
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
