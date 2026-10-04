"""Safe aggregate rank-movement audit of completed repeated-development systems."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest

BASE = Path("artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay")


def run(directory, variant, output):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if (
        report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Completed physical system required")
    baseline, actual = [
        json.loads((p / "selected_rankings.json").read_text())
        for p in (BASE, directory)
    ]
    summary = {}
    for mode, expected in (("unknown", 2000), ("known", 1340)):
        a = (
            pd.read_csv(BASE / f"{mode}_supported_leader_replay.csv")
            .set_index("key")
            .sort_index()
        )
        b = (
            pd.read_csv(directory / f"{mode}_{variant}.csv")
            .set_index("key")
            .sort_index()
        )
        if (
            not a.index.equals(b.index)
            or len(a) != expected
            or set(actual[mode]) != set(baseline[mode])
            or set(actual[mode]) != set(a.index)
        ):
            raise ValueError("Complete aligned original cohorts required")
        delta = (b.reciprocal_rank - a.reciprocal_rank).to_numpy()
        if not np.isfinite(delta).all():
            raise ValueError("Finite aligned RR differences required")
        rng = np.random.default_rng(20261003)
        bootstrap = np.concatenate(
            [
                delta[rng.integers(0, len(delta), (100, len(delta)))].mean(axis=1)
                for _ in range(100)
            ]
        )
        legacy_available = set(a.index[a.covered > 0])
        retained_truth = {key for key, ranks in actual[mode].items() if key in ranks}
        summary[mode] = {
            "molecules": expected,
            "legacy_candidate_availability_count": len(legacy_available),
            "truth_retained_in_full_list": len(retained_truth),
            "legacy_available_truth_not_retained": len(
                legacy_available - retained_truth
            ),
            "retained_truth_not_legacy_available": len(
                retained_truth - legacy_available
            ),
            "full_rank_changes": sum(
                actual[mode][k] != baseline[mode][k] for k in actual[mode]
            ),
            "first_identity_changes": sum(
                actual[mode][k][:1] != baseline[mode][k][:1] for k in actual[mode]
            ),
            "true_rr_improvements": int((delta > 0).sum()),
            "true_rr_regressions": int((delta < 0).sum()),
            "true_rr_unchanged": int((delta == 0).sum()),
            "new_true_top25": int(
                ((a.reciprocal_rank == 0) & (b.reciprocal_rank > 0)).sum()
            ),
            "lost_true_top25": int(
                ((a.reciprocal_rank > 0) & (b.reciprocal_rank == 0)).sum()
            ),
            "paired_mrr_gain": float(delta.mean()),
            "paired_bootstrap95_gain_interval": list(
                map(float, np.quantile(bootstrap, [0.025, 0.975]))
            ),
        }
    result = {
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "round": directory.name,
        "variant": variant,
        "baseline_rankings_sha256": digest(BASE / "selected_rankings.json"),
        "rankings_sha256": digest(directory / "selected_rankings.json"),
        "report_sha256": digest(directory / "report.json"),
        "bootstrap_seed": 20261003,
        "bootstrap_samples": 10000,
        "cohorts": summary,
        "scope": "Aggregate attribution on completed repeated development cohorts. Inherited covered flag is legacy candidate availability, not an audit of every current retrieval stage; retention is exact truth membership in full stored output list. Does not attribute exclusion to a specific gate. Bootstrap describes this cohort, not unseen-data acceptance. No per-molecule truth or ranking export, no new model or promotion.",
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
