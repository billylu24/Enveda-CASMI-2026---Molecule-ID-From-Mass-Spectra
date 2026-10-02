"""Matched checkpoint pilot against current selected decoder, fixed128 trajectories."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.generation_experiment import generate
from casmi_ml.generation_temperature_experiment import BASELINE, summarize
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ROOT, freeze


def run(output, checkpoint, limit=200):
    output, checkpoint = Path(output), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "checkpoint_sha256": digest(checkpoint),
            "baseline_sha256": digest(BASELINE),
            "sampling_source_sha256": digest(Path(generate.__code__.co_filename)),
            "samples": 128,
            "temperature": 0.8,
            "limit": limit,
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "cohort": "repeated_development",
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
        },
    )
    budget = StageBudget(output, "generation", "checkpoint_pilot", 86400)
    try:
        selected = generate(
            ROOT,
            checkpoint=checkpoint,
            samples=128,
            stable_sampling=True,
            limit=limit,
            deadline=budget.started + budget.allowance,
        )
    finally:
        budget.close()
    original = {r["key"]: r for r in json.loads(BASELINE.read_text())}
    result = {
        "diagnostic_only": True,
        "molecules": len(selected),
        "baseline": summarize([original[r["key"]] for r in selected]),
        "selected": summarize(selected),
        "independent_acceptance": False,
        "evaluation": "Fixed200 current-best checkpoint control; no submission gate",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.checkpoint, args.limit), indent=2))


if __name__ == "__main__":
    main()
