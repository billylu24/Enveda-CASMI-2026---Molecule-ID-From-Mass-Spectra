"""Complete catalog/window/current/scored-list funnel, labels used only in aggregates."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from casmi_ml.cached_structure_keys import structure_key
from casmi_ml.data import write_json
from casmi_ml.full_gap_candidate_system import LOCAL_CONTROL
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_runtime import resolve
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

OBSERVED = Path(
    "artifacts/research_loop/rounds/0403_conditioned_gate_impact_corrected_audit"
)
EXPECTED = Path(
    "artifacts/research_loop/rounds/0390_energy_conditioned_chemical_system"
)
BUCKETS = (
    "no_observable_query_rows",
    "original_first",
    "original_current_other",
    "external_scored",
    "valid_mass_window_unscored",
    "catalog_outside_valid_mass_window",
    "absent_external_catalog_and_original_current",
)


def bucket(truth, current, scored, window, catalog):
    if current and truth == current[0]:
        return "original_first"
    if truth in current:
        return "original_current_other"
    if truth in scored:
        return "external_scored"
    if truth in window:
        return "valid_mass_window_unscored"
    if truth in catalog:
        return "catalog_outside_valid_mass_window"
    return "absent_external_catalog_and_original_current"


def missing_query_keys(queries, expected, original):
    keys = set(queries)
    if keys - set(expected) or set(original) != set(expected):
        raise ValueError("Original full cohort differs")
    missing = set(expected) - keys
    if any(expected[k] or original[k] for k in missing):
        raise ValueError(
            "Missing query must have original and conditional empty rankings"
        )
    return missing


def rr(ranking, truth):
    return 1.0 / (ranking.index(truth) + 1) if truth in ranking[:25] else 0.0


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Local-only complete funnel diagnostic required")
    output = Path(output)
    report = json.loads((OBSERVED / "report.json").read_text())
    audit = json.loads((OBSERVED / "gate_impact_summary.json").read_text())
    selected = json.loads((OBSERVED / "selected_rankings.json").read_text())
    if (
        report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
        or not audit["all3340_conditional390_fullrank_exact"]
        or selected != json.loads((EXPECTED / "selected_rankings.json").read_text())
    ):
        raise ValueError("Source-bound complete exact390 observations required")
    config = json.loads((LOCAL_CONTROL / "external_config.json").read_text())
    catalog_path = Path(config["catalog"])
    if digest(catalog_path) != config["catalog_sha256"]:
        raise ValueError("Entire original external catalog required")
    factory = resolve(c, config["catalog_sha256"])
    if factory is None:
        raise ValueError(
            "Source-bound original complete packed candidate runtime required"
        )
    protocol = {
        "source_sha256": digest(__file__),
        "catalog_sha256": config["catalog_sha256"],
        "candidate_window_source_sha256": digest("casmi_ml/mass_candidates.py"),
        "observations_sha256": digest(OBSERVED / "private_score_observations.json"),
        "gate_protocol_sha256": digest(OBSERVED / "gate_impact_protocol.json"),
        "gate_report_sha256": digest(OBSERVED / "report.json"),
        "gate_summary_sha256": digest(OBSERVED / "gate_impact_summary.json"),
        "conditional_rankings_sha256": digest(OBSERVED / "selected_rankings.json"),
        "original_rankings_sha256": digest(LOCAL_CONTROL / "selected_rankings.json"),
        "missing_rows_policy": "Keep original full2000/1340 denominator. Queries with no supplied observable rows and empty original/conditional rankings remain explicit no_observable_query_rows bucket; do not invent mass windows or scorer evidence.",
        "candidate_runtime": c.read()["local_candidate_runtime"],
        "object_candidate_runtime": c.read().get("local_object_candidate_runtime"),
        "inputs": {
            m: {
                n: digest(CONTROL / m / n)
                for n in ("test.parquet", "baseline_full.json")
            }
            for m in ("unknown", "known")
        },
        "rule": "All original2000 unknown/1340known queries, entire original external catalog. "
        "Rebuild EXACT original charge-aware union candidate_window with unchanged valid "
        "fingerprints, no neural scoring, predictions or physical requests. Query molecule_id "
        "removed before window selection. Original current and source-bound403 actual scored "
        "lists unchanged. Truth enters only exclusive aggregate bucket and final rank statistics. "
        "Buckets prioritize originalfirst/currentother/scored/windowunscored/catalogoutwindow/absent. "
        "Windowunscored does NOT establish native top100 rejection causality. No shortlist "
        "extension, gate change, fitting, development model choice or deployment.",
        "local_only": True,
        "independent_acceptance": False,
    }
    freeze(output / "protocol.json", protocol)
    index = factory(pd.read_parquet(catalog_path))
    catalog_keys = set(index.catalog.inchikey14)
    observed = {
        (r["mode"], r["query"]): r
        for r in json.loads((OBSERVED / "private_score_observations.json").read_text())
    }
    original = json.loads((LOCAL_CONTROL / "selected_rankings.json").read_text())
    cohorts = {}
    for mode, total in (("unknown", 2000), ("known", 1340)):
        queries = pd.read_parquet(CONTROL / mode / "test.parquet")
        currents = {
            r["molecule_id"]: [structure_key(s) for s in r["smiles"]]
            for r in json.loads((CONTROL / mode / "baseline_full.json").read_text())
        }
        groups = queries.groupby("molecule_id", sort=False)
        if len(selected[mode]) != total:
            raise ValueError("Complete matched original query cohort required")
        missing = missing_query_keys(groups.groups, selected[mode], original[mode])
        if any(k in currents or (mode, k) in observed for k in missing):
            raise ValueError(
                "Missing query unexpectedly has original handoff/scorer input"
            )
        counts, by_bucket = Counter(), defaultdict(Counter)
        for key in missing:
            counts["molecules"] += 1
            counts["no_observable_query_rows"] += 1
            counts["truth_in_external_catalog"] += key in catalog_keys
            counts["original_current_empty"] += 1
            by_bucket["no_observable_query_rows"]["queries"] += 1
        for key, group in groups:
            if c.stopped():
                return None
            pool, _ = candidate_window(
                index, group.drop(columns="molecule_id"), "charge_aware_union"
            )
            window = set(pool.inchikey14)
            current = currents.get(key, [])
            record = observed.get((mode, key))
            scored = set(record["adjusted"]) if record else set()
            if record and (not current or record["first"] != current[0]):
                raise ValueError("Original first binding differs")
            if not (scored - {current[0]} if current else scored).issubset(window):
                raise ValueError(
                    "Observed external shortlist outside unchanged valid mass window"
                )
            name = bucket(key, current, scored, window, catalog_keys)
            values = by_bucket[name]
            values["queries"] += 1
            old_rr, new_rr = rr(original[mode][key], key), rr(selected[mode][key], key)
            values["original_final_top25"] += old_rr > 0
            values["conditional_final_top25"] += new_rr > 0
            values["rr_improved_queries"] += new_rr > old_rr
            values["rr_regressed_queries"] += new_rr < old_rr
            values["rr_delta_sum"] += new_rr - old_rr
            counts["molecules"] += 1
            counts["truth_in_external_catalog"] += key in catalog_keys
            counts["truth_in_valid_mass_window"] += key in window
            counts["truth_in_original_current"] += key in current
            counts["truth_in_actual_scored_list"] += key in scored
            counts["original_current_empty"] += not current
        if (
            counts["molecules"] != total
            or sum(v["queries"] for v in by_bucket.values()) != total
        ):
            raise ValueError("Exclusive complete funnel partition required")
        cohorts[mode] = {
            **dict(counts),
            "exclusive_buckets": {
                name: {
                    k: float(v) if k == "rr_delta_sum" else int(v)
                    for k, v in by_bucket[name].items()
                }
                for name in BUCKETS
            },
        }
        print(mode, json.dumps(cohorts[mode]), flush=True)
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "catalog_molecules": len(index.catalog),
        "original3340_predictions_unchanged": True,
        "new_physical_requests": 0,
        "cohorts": cohorts,
        "scope": "Repeated-development exact original external catalog/window/current/scored "
        "funnel only. External catalog coverage is not overall training/reference/generator "
        "coverage. Windowunscored is not a per-candidate rejection reason. No prediction, "
        "counterfactual accuracy, new acceptance cohort, tuning, activation or remote publication.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
