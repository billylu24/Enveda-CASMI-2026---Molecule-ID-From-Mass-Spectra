"""Complete label-free replay of the supported low leader on audited physical evidence."""

import argparse
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml.chembl_routed_inference import extend
from casmi_ml.data import write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.diagnostic_system_evaluation import BEST, resolve_local_incremental
from casmi_ml.low_fragment_first import CONTROL, promote_low_leader
from casmi_ml.low_leader_inference import LOW_LEADER_RULE
from casmi_ml.metfrag import digest
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

SCREEN = Path("artifacts/research_loop/rounds/0236_low_supported_leader_first")


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Replay is local-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    benchmark, inherited, runtime = resolve_local_incremental(str(CONTROL), None)
    if runtime is None:
        raise ValueError("Verified local incremental runtime required")
    control_report = json.loads((CONTROL / "report.json").read_text())
    if control_report.get("development_budget_fallbacks"):
        raise ValueError("Control physical evidence incomplete")
    config = json.loads((BEST / "external_config.json").read_text())
    config["low_fragment"] = LOW_LEADER_RULE
    expected = json.loads((CONTROL / "selected_rankings.json").read_text())
    screened = json.loads((SCREEN / "selected_rankings.json").read_text())
    engines = []

    def engine(*args, **kwargs):
        obj = DevelopmentIncrementalMetFrag(*args, benchmark=benchmark, **kwargs)
        engines.append(obj)
        return obj

    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "inference_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "low_leader_source_sha256": digest("casmi_ml/low_leader_inference.py"),
            "control_rankings_sha256": digest(CONTROL / "selected_rankings.json"),
            "screened_rankings_sha256": digest(SCREEN / "selected_rankings.json"),
            "inherited_failures_sha256": runtime["failures_sha256"],
            "config": config,
            "local_only": True,
            "independent_acceptance": False,
            "rule": "Independently construct control plus one supported low leader move from opaque routing status, then require actual unlabeled inference full-ranking equality for every original cohort molecule. Historical screen drift outside the low branch is separately attributed to the already audited physical control.",
            "input_sha256": {
                mode: {
                    name: digest(CONTROL / mode / name)
                    for name in (
                        "test.parquet",
                        "baseline.csv",
                        "baseline_full.json",
                        "routing.csv",
                    )
                }
                for mode in ("unknown", "known")
            },
        },
    )
    report, selected = {}, {}
    budget = 0
    for mode in ("unknown", "known"):
        if c.stopped():
            return None
        source, dest = CONTROL / mode, output / mode
        dest.mkdir(exist_ok=True)
        audit = pd.read_csv(source / "selected.csv.external.csv").set_index(
            "molecule_id"
        )
        original = {
            row["molecule_id"]: [structure_key(s) for s in row["smiles"]]
            for row in json.loads((source / "baseline_full.json").read_text())
        }
        target, promoted = {}, set()
        for key, ranking in expected[mode].items():
            target[key] = ranking
            if (
                key in audit.index
                and audit.loc[key, "status"] == "low_fragment_inserted"
            ):
                if audit.loc[key, "confidence"] >= 0.5:
                    raise ValueError("Supported low branch has high confidence")
                target[key] = promote_low_leader(original[key], ranking)
                promoted.add(key)
                if target[key] != screened[mode][key]:
                    raise ValueError("Supported group differs from screened winner")
        for name in (
            "test.parquet",
            "baseline.csv",
            "baseline_full.json",
            "routing.csv",
        ):
            shutil.copy2(source / name, dest / name)
        started = time.monotonic()
        with patch("casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=engine):
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
            row["molecule_id"]: [structure_key(s) for s in row["smiles"]]
            for row in json.loads((dest / "full.json").read_text())
        }
        actual.update({key: [] for key, ranking in target.items() if not ranking})
        if set(actual) != set(target):
            raise ValueError("Replay cohort differs")
        mismatches = [key for key in target if actual[key] != target[key]]
        write_json(dest / "private_mismatches.json", mismatches)
        execution = json.loads((dest / "selected.csv.external.report.json").read_text())
        budget += sum(
            n for status, n in execution["status_counts"].items() if "budget" in status
        )
        # Same candidate universe as the audited original model: promotion never
        # adds or removes a candidate. Copy its coverage, recompute ranks only.
        rows = pd.read_csv(CONTROL / f"{mode}_diagnostic_residual.csv").set_index("key")
        for key, ranking in actual.items():
            rank = ranking.index(key) + 1 if key in ranking[:25] else 0
            rows.loc[key, "reciprocal_rank"] = 1 / rank if rank else 0
            for n in (1, 5, 25):
                rows.loc[key, "top" + str(n)] = int(0 < rank <= n)
        from casmi_ml.low_fragment_first import metric_report

        observed_metrics = metric_report(rows)
        rows.reset_index().to_csv(
            output / f"{mode}_supported_leader_replay.csv", index=False
        )
        # Coverage for the screen itself uses this same unchanged candidate pool.
        corrected_screen = pd.read_csv(
            SCREEN / f"{mode}_low_supported_leader_first.csv"
        ).set_index("key")
        corrected_screen["covered"] = rows["covered"]
        report[mode] = {
            "molecules": len(target),
            "full_rank_matches": len(target) - len(mismatches),
            "promoted_groups": len(promoted),
            "metrics": observed_metrics,
            "corrected_screen_metrics": metric_report(corrected_screen),
            "historical_screen_outside_branch_drift": sum(
                actual[k] != screened[mode][k] for k in target if k not in promoted
            ),
            "seconds": execution["seconds"],
            "status_counts": execution["status_counts"],
            "parent_peak_rss_mib": execution["parent_peak_rss_mib"],
        }
        selected[mode] = actual
        print(mode, report[mode], flush=True)
        if mismatches:
            raise ValueError(
                "Complete label-free replay differs from independently derived ranking"
            )
    failures = set().union(*(e.failed_physical_requests for e in engines))
    write_json(output / "private_incremental_failures.json", sorted(failures))
    report.update(
        diagnostic_only=True,
        local_only=True,
        independent_acceptance=False,
        development_budget_fallbacks=budget,
        incremental_new_failure_contexts=len(failures - inherited),
        incremental_inherited_failure_contexts=len(failures & inherited),
        verified=budget == 0 and not (failures - inherited),
        resource_scope="Cached full development implementation replay; no cold inference or fresh holdout claim",
    )
    write_json(output / "selected_rankings.json", selected)
    write_json(output / "external_config.json", config)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
