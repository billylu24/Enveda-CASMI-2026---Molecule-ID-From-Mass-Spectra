"""Verify object storage for candidate frames against every current best rank."""

import argparse
import json
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.diagnostic_system_evaluation import resolve_local_incremental
from casmi_ml.local_verified_extension import verified_config
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.object_candidate_runtime import ObjectPackedCandidateIndex
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Sort amendment replay is local-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config, proof = verified_config(c)
    from casmi_ml import chembl_routed_inference as trial

    incumbent = c.root / "rounds" / proof["evidence_round"]
    runtime = c.read()["local_candidate_runtime"]
    benchmark_directory = c.root / "rounds/0271_object_candidate_window_benchmark"
    benchmark_report = json.loads((benchmark_directory / "report.json").read_text())
    if benchmark_report["full_value_and_float32_fingerprint_windows_exact"] != 200:
        raise ValueError("Exact candidate value/float32 benchmark required")
    benchmark, inherited, _ = resolve_local_incremental(str(CONTROL), None)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "inference_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "object_source_sha256": digest("casmi_ml/object_candidate_runtime.py"),
            "candidate_runtime": runtime,
            "benchmark_sha256": digest(benchmark_directory / "report.json"),
            "array_source_sha256": digest("casmi_ml/candidate_array_order.py"),
            "config": config,
            "incumbent_proof": proof,
            "local_only": True,
            "independent_acceptance": False,
            "rule": "Candidate string columns change only storage from Arrow to Python objects after original packed source/row identity validation. All values, order, bits, model and scores unchanged. All3340 complete rankings must match current verified best; original physical evidence and deadlines unchanged.",
        },
    )
    engines = []

    def engine(*args, **kwargs):
        obj = DevelopmentIncrementalMetFrag(*args, benchmark=benchmark, **kwargs)
        engines.append(obj)
        return obj

    expected = json.loads((incumbent / "selected_rankings.json").read_text())
    reports = {}
    selected = {}
    budgets = 0
    for mode in ["unknown", "known"]:
        if c.stopped():
            return None
        source = CONTROL / mode
        dest = output / mode
        dest.mkdir(exist_ok=True)
        for name in [
            "test.parquet",
            "baseline.csv",
            "baseline_full.json",
            "routing.csv",
        ]:
            shutil.copy2(source / name, dest / name)
        started = time.monotonic()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=engine
                )
            )
            stack.enter_context(
                patch.object(
                    trial,
                    "CandidateIndex",
                    new=lambda catalog: ObjectPackedCandidateIndex(
                        catalog, runtime["store"], config["catalog_sha256"]
                    ),
                )
            )
            if config.get("query_residual"):
                from casmi_ml.diagnostic_query_ranker import (
                    load_query_ranker,
                    score_chemical_group,
                )

                query_ranker, _ = load_query_ranker(
                    config["critic"], config["query_residual"]["original_critic"]
                )
                stack.enter_context(
                    patch.object(
                        trial,
                        "DirectRanker",
                        new=lambda architecture, ranker=query_ranker: ranker,
                    )
                )
                stack.enter_context(
                    patch.object(
                        trial,
                        "score_group",
                        new=score_chemical_group,
                    )
                )
            trial.extend(
                dest / "test.parquet",
                dest / "baseline.csv",
                dest / "baseline_full.json",
                dest / "routing.csv",
                dest / "selected.csv",
                Path("."),
                config,
                deadline=started + 3600,
                fragment_deadline=started + 1200,
                java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
                cache="artifacts/chemistry_20261001/chembl_metfrag_cache",
                output_full_rankings=dest / "full.json",
                reference=ReferenceIndex(
                    "artifacts/research_loop/rounds/0001_mass_v2/reference/" + mode
                ),
            )
        actual = {
            row["molecule_id"]: [structure_key(s) for s in row["smiles"]]
            for row in json.loads((dest / "full.json").read_text())
        }
        actual.update(
            {key: [] for key, ranking in expected[mode].items() if not ranking}
        )
        if set(actual) != set(expected[mode]):
            raise ValueError("Object storage original cohort differs")
        mismatches = [key for key in actual if actual[key] != expected[mode][key]]
        write_json(dest / "private_mismatches.json", mismatches)
        if mismatches:
            raise ValueError(
                "Object candidate runtime ranks differ; retain original runtime"
            )
        execution = json.loads((dest / "selected.csv.external.report.json").read_text())
        budgets += sum(
            n for status, n in execution["status_counts"].items() if "budget" in status
        )
        reports[mode] = {
            "molecules": len(actual),
            "full_rank_matches": len(actual),
            "seconds": execution["seconds"],
            "parent_peak_rss_mib": execution["parent_peak_rss_mib"],
        }
        selected[mode] = actual
        print(mode, reports[mode], flush=True)
    failures = set().union(*(engine.failed_physical_requests for engine in engines))
    reports.update(
        diagnostic_only=True,
        local_only=True,
        independent_acceptance=False,
        development_budget_fallbacks=budgets,
        incremental_new_failure_contexts=len(failures - inherited),
        verified=not budgets and not (failures - inherited),
    )
    write_json(output / "private_incremental_failures.json", sorted(failures))
    write_json(output / "selected_rankings.json", selected)
    write_json(output / "external_config.json", config)
    write_json(output / "report.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
