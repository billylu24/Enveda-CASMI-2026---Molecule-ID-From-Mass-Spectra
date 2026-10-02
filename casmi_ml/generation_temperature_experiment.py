"""Fixed-trajectories temperature pilot against the selected decoder cache."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.generation_experiment import generate
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ROOT, freeze

CHECKPOINT = Path("artifacts/research_loop/rounds/0020_decoder_single_control/model.pt")
BASELINE = (
    ROOT / "generation/researchdev_samples128_limitall_stable_v2_98194888a437.json"
)


def summarize(rows):
    result = {
        name: sum(r["statistics"][name] for r in rows)
        for name in [
            "samples",
            "terminated",
            "valid",
            "mass_matching",
            "unique_mass_matching",
        ]
    }
    result.update(
        queries_with_candidates=sum(bool(r["candidates"]) for r in rows),
        exact_structure_hits=sum(
            any(c["key"] == r["key"] for c in r["candidates"]) for r in rows
        ),
    )
    return result


def run(output, temperature, limit=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "generator_sha256": digest(CHECKPOINT),
            "baseline_sha256": digest(BASELINE),
            "sampling_source_sha256": digest(Path(generate.__code__.co_filename)),
            "temperature": temperature,
            "baseline_temperature": 0.8,
            "samples": 128,
            "limit": limit,
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
        },
    )
    budget = StageBudget(output, "generation", "temperature_sampling", 86400)
    try:
        generated = generate(
            ROOT,
            checkpoint=CHECKPOINT,
            samples=128,
            stable_sampling=True,
            limit=limit,
            deadline=budget.started + budget.allowance,
            temperature=temperature,
        )
    finally:
        budget.close()
    original = {r["key"]: r for r in json.loads(BASELINE.read_text())}
    result = {
        "molecules": len(generated),
        "baseline": summarize([original[r["key"]] for r in generated]),
        "selected": summarize(generated),
        "diagnostic_only": True,
        "temperature": temperature,
        "independent_acceptance": False,
        "evaluation": "Fixed development pilot; exact structure hits, not a submission gate",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.temperature, args.limit), indent=2))


if __name__ == "__main__":
    main()
