"""Full paired local inference for the candidate-facing diagnostic query residual."""

import argparse
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.chembl_routed_inference import extend
from casmi_ml.data import write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.diagnostic_query_ranker import load_query_ranker, score_chemical_group
from casmi_ml.diagnostic_system_evaluation import resolve_local_incremental
from casmi_ml.low_fragment_first import CONTROL, metric_report
from casmi_ml.metfrag import digest
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

LOCAL_CONTROL = Path(
    "artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay"
)


def run(output, critic, control=False):
    c = Controller()
    if not c.local_only():
        raise ValueError("Query residual evaluation is local-only")
    output, critic = Path(output), Path(critic)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((LOCAL_CONTROL / "external_config.json").read_text())
    original_critic = config["critic"]
    ranker, weights = load_query_ranker(critic, original_critic)
    if control and any(
        bool(t.count_nonzero())
        for name, t in weights["state_dict"].items()
        if name
        in ["query_residual.correction.2.weight", "query_residual.correction.2.bias"]
    ):
        raise ValueError("Implementation control requires strictly zero residual")
    config.update(critic=str(critic), critic_sha256=digest(critic))
    config["query_residual"] = {
        "original_critic": original_critic,
        "original_critic_sha256": digest(original_critic),
        "ranker_source_sha256": digest("casmi_ml/diagnostic_query_ranker.py"),
        "residual_source_sha256": digest("casmi_ml/diagnostic_query_residual.py"),
    }
    benchmark, inherited, _ = resolve_local_incremental(str(CONTROL), None)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "inference_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "query_source_sha256": digest("casmi_ml/diagnostic_query_ranker.py"),
            "query_residual_sha256": digest("casmi_ml/diagnostic_query_residual.py"),
            "critic_sha256": digest(critic),
            "original_critic_sha256": digest(original_critic),
            "paired_rankings_sha256": digest(LOCAL_CONTROL / "selected_rankings.json"),
            "control": control,
            "local_only": True,
            "independent_acceptance": False,
            "config": config,
            "rule": "Same complete original physical baseline and verified low-leader/highstrong-slot rule; only frozen original critic normalized query gains precise chemical/contextual residual. Same original encoder probabilities, candidates, mass windows, generation and references; critic molecular tower and scale unchanged. Zero query residual full-ranking control required before training arm evaluation.",
            "scope": "Cached complete repeated development comparison, no fresh holdout or cold resource proof",
        },
    )
    engines = []

    def engine(*args, **kwargs):
        obj = DevelopmentIncrementalMetFrag(*args, benchmark=benchmark, **kwargs)
        engines.append(obj)
        return obj

    def constructor(architecture):
        if architecture != "fingerprint":
            raise ValueError("Original critic architecture required")
        return ranker

    expected = json.loads((LOCAL_CONTROL / "selected_rankings.json").read_text())
    selected, report = {}, {}
    budget = 0
    for mode in ("unknown", "known"):
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
        with (
            patch("casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=engine),
            patch("casmi_ml.chembl_routed_inference.DirectRanker", new=constructor),
            patch(
                "casmi_ml.chembl_routed_inference.score_group", new=score_chemical_group
            ),
        ):
            extend(
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
            r["molecule_id"]: [structure_key(s) for s in r["smiles"]]
            for r in json.loads((dest / "full.json").read_text())
        }
        actual.update(
            {key: [] for key, ranking in expected[mode].items() if not ranking}
        )
        if set(actual) != set(expected[mode]):
            raise ValueError("Full query system cohort differs")
        if control:
            mismatches = [key for key in actual if actual[key] != expected[mode][key]]
            write_json(dest / "private_mismatches.json", mismatches)
            if mismatches:
                raise ValueError(
                    "Zero query residual full rankings differ from verified actual inference"
                )
        rows = pd.read_csv(
            LOCAL_CONTROL / f"{mode}_supported_leader_replay.csv"
        ).set_index("key")
        for key, ranking in actual.items():
            rank = ranking.index(key) + 1 if key in ranking[:25] else 0
            rows.loc[key, "reciprocal_rank"] = 1 / rank if rank else 0
            for n in (1, 5, 25):
                rows.loc[key, "top" + str(n)] = int(0 < rank <= n)
        if not np.isfinite(rows.reciprocal_rank).all():
            raise ValueError("Invalid query metrics")
        report[mode] = {"chemical_query_residual": metric_report(rows)}
        rows.reset_index().to_csv(
            output / f"{mode}_chemical_query_residual.csv", index=False
        )
        selected[mode] = actual
        execution = json.loads((dest / "selected.csv.external.report.json").read_text())
        budget += sum(
            n for status, n in execution["status_counts"].items() if "budget" in status
        )
        print(mode, report[mode], flush=True)
    failures = set().union(*(engine.failed_physical_requests for engine in engines))
    write_json(output / "private_incremental_failures.json", sorted(failures))
    report.update(
        diagnostic_only=control or budget > 0 or bool(failures - inherited),
        local_only=True,
        independent_acceptance=False,
        development_budget_fallbacks=budget,
        incremental_new_failure_contexts=len(failures - inherited),
        incremental_inherited_failure_contexts=len(failures & inherited),
        control_passed=control and budget == 0 and not (failures - inherited),
    )
    old = json.loads((CONTROL / "report.json").read_text())
    local = json.loads((LOCAL_CONTROL / "report.json").read_text())
    report["required_comparators"] = {
        "completed_original": {
            "directory": str(CONTROL),
            "variant": "diagnostic_residual",
            "metrics": {m: old[m]["diagnostic_residual"] for m in ("unknown", "known")},
        },
        "verified_low_leader": {
            "directory": str(LOCAL_CONTROL),
            "variant": "supported_leader_replay",
            "metrics": {m: local[m]["metrics"] for m in ("unknown", "known")},
        },
    }
    write_json(output / "external_config.json", config)
    write_json(output / "selected_rankings.json", selected)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--control", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
