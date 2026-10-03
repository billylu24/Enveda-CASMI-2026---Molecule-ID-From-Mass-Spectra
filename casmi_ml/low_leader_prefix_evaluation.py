"""Preregistered stronger original-prefix evidence for low-route first-slot promotion."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.diagnostic_system_evaluation import BEST, resolve_local_incremental
from casmi_ml.low_fragment_first import CONTROL, metric_report, promote_low_leader
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def supported_over_prefix(leader, original, fragments, fallback):
    prefix = original[:3]
    return bool(
        prefix
        and not fallback
        and leader in fragments
        and all(key in fragments for key in prefix)
        and fragments[leader] > max(fragments[key] for key in prefix)
    )


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Evaluation is local-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best = c.read()["development_best"]
    if best["round"] != "0236_low_supported_leader_first":
        raise ValueError("Verified low leader incumbent required")
    verified = c.read().get("local_verified_inference", {})
    if not verified:
        raise ValueError(
            "First-slot rule requires complete prior implementation replay"
        )
    benchmark, inherited, _ = resolve_local_incremental(str(CONTROL), None)
    config = json.loads((BEST / "external_config.json").read_text())
    high = config["high_fragment"]
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "control_rankings_sha256": digest(CONTROL / "selected_rankings.json"),
            "incumbent_directory": best["directory"],
            "incumbent_variant": best["variant"],
            "incumbent_report_sha256": digest(Path(best["directory"]) / "report.json"),
            "rule": "Only control low_fragment_inserted groups under unchanged fragment informativeness and criticfirst+.05 gates; additionally require leader raw fragment score strictly exceeds every actual original first3 score. Missing prefix evidence or budget fallback keeps complete control insertion. No tuning of thresholds or labels for group selection; all original2000/1340 molecules retained.",
            "new_training": False,
            "new_candidates": False,
            "local_only": True,
            "independent_acceptance": False,
            "fragment_seconds_per_mode": 1200,
        },
    )
    engine = DevelopmentIncrementalMetFrag(
        config["jar"],
        "artifacts/chemistry_20261001/chembl_metfrag_cache",
        java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
        classes=Path(high["worker_class"]).parent,
        threads=2,
        benchmark=benchmark,
    )
    selected = json.loads((CONTROL / "selected_rankings.json").read_text())
    reports, counts = {}, {}
    try:
        for mode in ("unknown", "known"):
            source = CONTROL / mode
            audit = pd.read_csv(source / "selected.csv.external.csv").set_index(
                "molecule_id"
            )
            raw = {
                row["molecule_id"]: row["smiles"]
                for row in json.loads((source / "baseline_full.json").read_text())
            }
            structures = {
                row["molecule_id"]: row["smiles"]
                for row in json.loads((source / "full.json").read_text())
            }
            test = pd.read_parquet(source / "test.parquet")
            groups = test.groupby("molecule_id", sort=False).indices
            deadline = time.monotonic() + 1200
            accepted = fallbacks = missing = eligible = 0
            for key, order in selected[mode].items():
                if (
                    key not in audit.index
                    or audit.loc[key, "status"] != "low_fragment_inserted"
                ):
                    continue
                eligible += 1
                original = [structure_key(s) for s in raw[key]]
                first_slot = promote_low_leader(original, order)
                leader = first_slot[0]
                lookup = {structure_key(s): s for s in structures[key]}
                # Original candidate representation is pinned to baseline SMILES,
                # even if equivalent catalog SMILES use a different notation.
                lookup.update({structure_key(s): s for s in raw[key]})
                fragments, fallback = score_group_merged(
                    engine,
                    test.iloc[groups[key]]
                    .drop(columns=["molecule_id"])
                    .to_dict("records"),
                    {k: lookup[k] for k in [leader] + original[:3]},
                    deadline=deadline,
                )
                fallbacks += int(fallback)
                missing += int(any(k not in fragments for k in [leader] + original[:3]))
                if supported_over_prefix(leader, original, fragments, fallback):
                    selected[mode][key] = first_slot
                    accepted += 1
            counts[mode] = {
                "eligible": eligible,
                "prefix_supported": accepted,
                "budget_fallbacks": fallbacks,
                "missing_prefix_scores": missing,
            }
            rows = pd.read_csv(CONTROL / f"{mode}_diagnostic_residual.csv").set_index(
                "key"
            )
            for key, order in selected[mode].items():
                rank = order.index(key) + 1 if key in order[:25] else 0
                rows.loc[key, "reciprocal_rank"] = 1 / rank if rank else 0
                for n in (1, 5, 25):
                    rows.loc[key, "top" + str(n)] = int(0 < rank <= n)
            reports[mode] = {"prefix_supported_leader": metric_report(rows)}
            rows.reset_index().to_csv(
                output / f"{mode}_prefix_supported_leader.csv", index=False
            )
            print(mode, reports[mode], counts[mode], flush=True)
    finally:
        engine.close()
    failures = engine.failed_physical_requests
    write_json(output / "private_incremental_failures.json", sorted(failures))
    reports.update(
        local_only=True,
        independent_acceptance=False,
        diagnostics=counts,
        development_budget_fallbacks=sum(
            x["budget_fallbacks"] for x in counts.values()
        ),
        incremental_new_failure_contexts=len(failures - inherited),
    )
    reports["diagnostic_only"] = bool(
        reports["development_budget_fallbacks"]
        or reports["incremental_new_failure_contexts"]
    )
    control_report = json.loads((CONTROL / "report.json").read_text())
    reports["required_comparators"] = {
        "completed_evidence_original_model": {
            "directory": str(CONTROL),
            "variant": "diagnostic_residual",
            "metrics": {
                m: control_report[m]["diagnostic_residual"]
                for m in ("unknown", "known")
            },
        }
    }
    write_json(output / "selected_rankings.json", selected)
    write_json(output / "report.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
