"""Aggregate original platform/local request cost observations, without inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

PLATFORM = Path(
    "artifacts/research_loop/overlap_request_runtime_platform_progress/runtime_diagnostic/request_events.json"
)
LOCAL = Path(
    "artifacts/research_loop/rounds/0407_overlap_request_local400_observation/request_events.json"
)


def summary(rows):
    durations = np.asarray([r["seconds"] for r in rows])
    ratios = np.asarray([r["seconds"] / r["local_seconds"] for r in rows])
    return {
        "requests": len(rows),
        "platform_seconds_sum": float(durations.sum()),
        "local_same_request_seconds_sum": float(sum(r["local_seconds"] for r in rows)),
        "platform_duration_quantiles": np.quantile(
            durations, [0, 0.5, 0.9, 0.99, 1]
        ).tolist()
        if len(rows)
        else [],
        "platform_over_local_ratio_quantiles": np.quantile(
            ratios, [0, 0.5, 0.9, 0.99, 1]
        ).tolist()
        if len(rows)
        else [],
    }


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Local-only runtime observation audit required")
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "platform_events_sha256": digest(PLATFORM),
            "local_events_sha256": digest(LOCAL),
            "rule": "Original unchanged400 platform diagnosis only. Keep complete paired original request digests, require candidate/peak counts exact. Prespecified observed-start-before/after377.277425232 generation end, first-after-JVM-start(worker_requests1), candidate bins<=25/26..100/>100, peaks<=100/101..300/301..600/>600. Descriptive overlapping request costs, not wall time, causality, matched hardware or optimization speed proof. No requests rerun, no settings/model/deadline modification. No labels or request IDs exported.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    local = {r["request_digest"]: r for r in json.loads(LOCAL.read_text())}
    platform = json.loads(PLATFORM.read_text())
    rows = []
    for r in platform:
        if r["status"] != "complete":
            continue
        other = local[r["request_digest"]]
        if other["status"] != "complete" or any(
            r[k] != other[k] for k in ("candidates", "peaks", "candidate_threads")
        ):
            raise ValueError("Original paired request dimensions differ")
        rows.append(dict(r, local_seconds=other["seconds"]))
    if len(rows) != 227:
        raise ValueError("Complete original227 paired requests required")
    bins = {
        "start_during_generation": [
            r for r in rows if r["started_seconds"] < 377.277425232
        ],
        "start_after_generation": [
            r for r in rows if r["started_seconds"] >= 377.277425232
        ],
        "first_after_jvm_start": [r for r in rows if r["worker_requests"] == 1],
        "other_jvm_requests": [r for r in rows if r["worker_requests"] != 1],
        "candidates_le25": [r for r in rows if r["candidates"] <= 25],
        "candidates_26_100": [r for r in rows if 25 < r["candidates"] <= 100],
        "candidates_gt100": [r for r in rows if r["candidates"] > 100],
    }
    for lo, hi in ((0, 100), (100, 300), (300, 600), (600, float("inf"))):
        bins[f"peaks_{lo + 1}_{hi}"] = [r for r in rows if lo < r["peaks"] <= hi]
    correlations = {}
    for name, values in {
        "peak_count": [r["peaks"] for r in rows],
        "candidate_count": [r["candidates"] for r in rows],
        "local_cost": [r["local_seconds"] for r in rows],
    }.items():
        correlations[name] = float(
            spearmanr(values, [r["seconds"] for r in rows]).statistic
        )
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "overall": summary(rows),
        "prespecified_bins": {name: summary(values) for name, values in bins.items()},
        "observational_spearman_duration": correlations,
        "scope": "Original failed acceptance diagnosis request cost observations. Bins have different inputs and workload; during/after generation or JVM-start association is not CPU contention causality. Sums overlap and are not total inference time. No timeout extension, request omission, physical/model/recipe changes or deployment.",
    }
    write_json(output / "report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
