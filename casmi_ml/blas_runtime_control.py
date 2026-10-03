"""Full current-best local ranking verification under single-thread BLAS."""

import argparse
import json
import shutil
from pathlib import Path

from threadpoolctl import threadpool_limits

from casmi_ml.data import write_json
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.local_verified_extension import run as verified_run
from casmi_ml.local_verified_extension import verified_config
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

BENCHMARK = Path("artifacts/research_loop/rounds/0311_blas_candidate_score_benchmark")


def run(output):
    c = Controller()
    config, proof = verified_config(c)
    benchmark = json.loads((BENCHMARK / "report.json").read_text())
    if benchmark["windows"] != 200 or benchmark["candidate_ranking_mismatches"]:
        raise ValueError("Complete real score-window ranking control required")
    timings = benchmark["score_seconds_two_repetitions"]
    if timings["blas1"] >= 0.9 * timings["blas24"]:
        raise ValueError("No clear BLAS score microbenchmark improvement")
    output = Path(output)
    reference = c.root / "rounds" / proof["evidence_round"]
    expected = json.loads((reference / "selected_rankings.json").read_text())
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "entry_source_sha256": digest("casmi_ml/local_verified_extension.py"),
            "benchmark_report_sha256": digest(BENCHMARK / "report.json"),
            "baseline_proof": proof,
            "baseline_rankings_sha256": digest(reference / "selected_rankings.json"),
            "rule": "Only BLAS user_api concurrency24->1; torch4 and scientific graph/candidates/ordering/physical routing unchanged. Exact complete current-best rankings required on original2000 unknown and1340 known.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    selected, report = {}, {}
    budget = new_failures = 0
    for mode, count in [("unknown", 2000), ("known", 1340)]:
        if c.stopped():
            return None
        dest = output / mode
        dest.mkdir(parents=True, exist_ok=True)
        for name in [
            "test.parquet",
            "baseline.csv",
            "baseline_full.json",
            "routing.csv",
        ]:
            shutil.copy2(CONTROL / mode / name, dest / name)
        with threadpool_limits(limits=1, user_api="blas"):
            verified_run(
                dest / "test.parquet",
                dest / "baseline.csv",
                dest / "baseline_full.json",
                dest / "routing.csv",
                Path("artifacts/research_loop/rounds/0001_mass_v2/reference") / mode,
                dest / "selected.csv",
            )
        actual = {
            r["molecule_id"]: [structure_key(s) for s in r["smiles"]]
            for r in json.loads((dest / "selected.csv.full.json").read_text())
        }
        actual.update({k: [] for k, v in expected[mode].items() if not v})
        if len(actual) != count or set(actual) != set(expected[mode]):
            raise ValueError("BLAS full control cohort differs")
        mismatches = [k for k in actual if actual[k] != expected[mode][k]]
        write_json(dest / "private_mismatches.json", mismatches)
        if mismatches:
            raise ValueError(
                "Single-thread BLAS full ranks differ from current-best evidence"
            )
        detail = json.loads((dest / "selected.csv.external.report.json").read_text())
        complete = json.loads((dest / "selected.csv.local_verified.json").read_text())
        budget += complete["development_budget_fallbacks"]
        new_failures += complete["new_failure_contexts"]
        selected[mode] = actual
        report[mode] = {
            "molecules": count,
            "full_rank_matches": count,
            "seconds": detail["seconds"],
            "parent_peak_rss_mib": detail["parent_peak_rss_mib"],
            "status_counts": detail["status_counts"],
        }
        print(mode, report[mode], flush=True)
    report.update(
        diagnostic_only=True,
        local_only=True,
        independent_acceptance=False,
        development_budget_fallbacks=budget,
        incremental_new_failure_contexts=new_failures,
        verified=not budget and not new_failures,
        scope="Exact current-best full cached ranking control. Microbenchmark supports score-segment timing; this unpaired full control does not prove cold end-to-end speed gain.",
    )
    write_json(output / "report.json", report)
    write_json(output / "selected_rankings.json", selected)
    write_json(output / "external_config.json", config)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
