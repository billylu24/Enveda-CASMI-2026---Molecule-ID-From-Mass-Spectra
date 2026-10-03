"""Verify an explicit runtime-only sort amendment against every incumbent ranking."""

import argparse
import importlib.util
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
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_runtime import resolve
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

AMENDMENT = Path("artifacts/research_loop/array_order_source_amendment.json")


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Sort amendment replay is local-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    proof = c.read()["local_verified_inference"]
    amendment = json.loads(AMENDMENT.read_text())
    if (
        proof["inference_sha256"] != amendment["original_inference_sha256"]
        or digest(amendment["amended_source"]) != amendment["amended_inference_sha256"]
    ):
        raise ValueError(
            "Sort amendment does not bind current verified original and amended source"
        )
    if (
        digest("casmi_ml/chembl_routed_inference.py")
        != amendment["original_inference_sha256"]
    ):
        raise ValueError("Original inference source changed before array replay")
    module_spec = importlib.util.spec_from_file_location(
        "casmi_array_runtime_verified_trial", amendment["amended_source"]
    )
    trial = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(trial)
    incumbent = c.root / "rounds" / proof["evidence_round"]
    for name, field in [
        ("report.json", "replay_report_sha256"),
        ("selected_rankings.json", "rankings_sha256"),
        ("external_config.json", "external_config_sha256"),
    ]:
        if digest(incumbent / name) != proof[field]:
            raise ValueError("Sort amendment incumbent changed: " + name)
    config = json.loads((incumbent / "external_config.json").read_text())
    config["candidate_order_runtime"] = "array_v1"
    benchmark, inherited, _ = resolve_local_incremental(str(CONTROL), None)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "source_amendment": amendment,
            "array_source_sha256": digest("casmi_ml/candidate_array_order.py"),
            "config": config,
            "incumbent_proof": proof,
            "local_only": True,
            "independent_acceptance": False,
            "rule": "Only conditional high-route score/key ordering moves from pandas scalar row access to array lexsort. Complete original3340 cohort rankings must exactly match currently verified incumbent. No query label selection, candidate or score arithmetic changes; same model and physical evidence.",
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
            candidate_factory = resolve(c, config["catalog_sha256"])
            if candidate_factory is not None:
                stack.enter_context(
                    patch.object(
                        trial,
                        "CandidateIndex",
                        new=candidate_factory,
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
            raise ValueError("Sort amendment original cohort differs")
        mismatches = [key for key in actual if actual[key] != expected[mode][key]]
        write_json(dest / "private_mismatches.json", mismatches)
        if mismatches:
            raise ValueError(
                "Array runtime complete ranks differ; retain original inference source"
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


def activate(c, directory):
    if not c.local_only():
        raise ValueError("Array runtime activation is local-only")
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    proof = c.read()["local_verified_inference"]
    if not report["verified"] or proof != protocol["incumbent_proof"]:
        raise ValueError("Complete unchanged-incumbent array replay required")
    if (
        digest(protocol["source_amendment"]["amended_source"])
        != protocol["source_amendment"]["amended_inference_sha256"]
        or digest("casmi_ml/candidate_array_order.py")
        != protocol["array_source_sha256"]
    ):
        raise ValueError("Array inference source changed")
    if any(
        report[mode]["full_rank_matches"] != count
        for mode, count in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("All3340 array ranks required")
    source_amendment = protocol["source_amendment"]
    source = Path("casmi_ml/chembl_routed_inference.py")
    if (
        digest(source) != source_amendment["original_inference_sha256"]
        or digest(source_amendment["amended_source"])
        != source_amendment["amended_inference_sha256"]
    ):
        raise ValueError("Original or trial source changed before array activation")
    source.write_bytes(Path(source_amendment["amended_source"]).read_bytes())
    new = {
        **proof,
        "evidence_round": directory.name,
        "replay_report_sha256": digest(directory / "report.json"),
        "rankings_sha256": digest(directory / "selected_rankings.json"),
        "external_config_sha256": digest(directory / "external_config.json"),
        "inference_sha256": protocol["source_amendment"]["amended_inference_sha256"],
        "array_source_sha256": protocol["array_source_sha256"],
        "scope": "Same incumbent score arithmetic/models and every3340 full rank; array score/key ordering runtime amendment only; cached replay, no accuracy or cold acceptance gain",
    }
    c.change(
        lambda state: state.update(
            local_verified_inference=new,
            local_array_runtime={
                "enabled": True,
                "local_only": True,
                "source_sha256": protocol["array_source_sha256"],
                "control_directory": str(directory),
            },
        )
    )
    write_json("results/research_loop/array_order_runtime_activation.json", new)
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
