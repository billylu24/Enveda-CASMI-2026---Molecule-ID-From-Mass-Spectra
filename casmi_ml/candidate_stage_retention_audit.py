"""Trace frozen retrieval pools and bounded generation into the original handoff."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.generated_first_critic import promotion_prefix
from casmi_ml.generated_score_combination import (
    GENERATED,
    SCORES,
    SOURCE,
    combined_order,
)
from casmi_ml.generated_second_reference import select_prefix
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT

HANDOFF = Path(
    "artifacts/research_loop/rounds/0149_chembl_routed_combination/low_rankings.json"
)
LEGACY = Path("artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay")
PROPOSAL_PROTOCOL = Path(
    "artifacts/research_loop/rounds/0206_full181_audited_exact_control/protocol.json"
)
PAIRS = Path("artifacts/research_loop/rounds/0061_generated_first_critic/scores.json")


def run(output):
    if not Controller().local_only():
        raise ValueError("Stage diagnosis requires persistent local-only mode")
    proposal_path = Path(
        "artifacts/research_loop/rounds/0147_chembl_all_query_proposals/corrected_only_proposals.json"
    )
    proof = json.loads(PROPOSAL_PROTOCOL.read_text())
    if digest(proposal_path) != proof["proposal_sha256"]:
        raise ValueError("External proposal binding differs")
    proposals = json.loads(proposal_path.read_text())
    native_path = Path(
        "artifacts/research_loop/rounds/0071_chembl_candidate_slots/proposals.json"
    )
    native = json.loads(native_path.read_text())
    final = json.loads((LEGACY / "selected_rankings.json").read_text())
    stored = json.loads(HANDOFF.read_text())["baseline"]
    generated = {r["key"]: r["candidates"] for r in json.loads(GENERATED.read_text())}
    scores, pairs = json.loads(SCORES.read_text()), json.loads(PAIRS.read_text())
    if len(generated) != 2000:
        raise ValueError("Complete frozen generation required")
    ordered = {
        k: combined_order(v, scores.get(k, {}).get("fingerprint", {}), (1.0, 0.0, 0.5))
        for k, v in generated.items()
    }
    report, inputs = {}, {}
    for mode, expected in (("unknown", 2000), ("known", 1340)):
        records = SOURCE / f"{mode}_records.json"
        conf_file = ROOT / f"researchdev_{mode}_chemical.json"
        reference_file = (
            Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
            / mode
            / "rows.parquet"
        )
        legacy_file = LEGACY / f"{mode}_supported_leader_replay.csv"
        rows = json.loads(records.read_text())
        confidence = {
            r["key"]: r["confidence"] for r in json.loads(conf_file.read_text())
        }
        observed = set(
            pd.read_parquet(reference_file, columns=["inchikey14"]).inchikey14
        )
        legacy = pd.read_csv(legacy_file).set_index("key")
        selected = [r for r in rows if mode == "unknown" or r["known"]]
        keys = {r["key"] for r in selected}
        if (
            len(selected) != expected
            or len(keys) != expected
            or keys != set(legacy.index)
            or keys != set(stored[mode])
        ):
            raise ValueError("Complete aligned cohorts required")
        audit_path = LEGACY / mode / "selected.csv.external.csv"
        audit = pd.read_csv(audit_path).set_index("molecule_id")
        counts, missing, final_missing = Counter(), Counter(), Counter()
        for row in selected:
            key = row["key"]
            base = row["variants"]["baseline"]
            protected = protects_reference(base["ranking"], observed, confidence[key])
            chosen = base if protected else row["variants"]["expansion_1"]
            rank, pool, gen = chosen["ranking"], chosen["pool"], ordered[key]
            prefix = (
                select_prefix(
                    base["ranking"], observed, confidence[key], "second_unreferenced"
                )
                if protected
                else 2
            )
            novel = [k for k in gen if k not in set(rank)]
            if confidence[key] < 0.2 and rank and novel:
                prefix = promotion_prefix(
                    rank, gen, pairs[key + ":" + rank[0] + ":" + novel[0]], prefix, 0.05
                )
            actual = insert_generated(rank, gen, prefix, 5)
            if actual != stored[mode][key]:
                raise ValueError(
                    "Reconstructed original handoff must be full-rank exact"
                )
            available = key in pool or key in gen
            inherited_available = bool(legacy.loc[key, "covered"])
            if inherited_available != (key in actual or key in proposals[key]):
                raise ValueError(
                    "Legacy availability must match original handoff plus all proposals"
                )
            counts["truth_in_all_external_proposals"] += key in proposals[key]
            counts["truth_in_first100_novel_external_proposals"] += (
                key in [k for k in proposals[key] if k not in set(actual)][:100]
            )
            counts["selected_pool_or_generated_available"] += available
            counts["legacy_available_outside_selected_pool_and_generated"] += (
                inherited_available and not available
            )
            counts["selected_pool_or_generated_outside_legacy_available"] += (
                available and not inherited_available
            )
            route_candidates = proposals[key] if confidence[key] >= 0.5 else native[key]
            shortlist = [k for k in route_candidates if k not in set(actual)][:100]
            counts["truth_in_actual_route_first100_novel_external"] += key in shortlist
            if inherited_available and key not in final[mode][key]:
                if key not in shortlist:
                    final_missing["not_in_actual_route_first100_novel_external"] += 1
                else:
                    final_missing[
                        "within_first100_status_" + audit.loc[key, "status"]
                    ] += 1
            counts["full_handoff_matches"] += 1
            counts["reference_protected_queries"] += protected
            counts["truth_in_selected_retrieval_pool"] += key in pool
            counts["truth_in_selected_retrieval_ranking"] += key in rank
            counts["truth_in_all_generated_candidates"] += key in gen
            counts["truth_in_first_five_novel_generated"] += key in novel[:5]
            counts["truth_added_by_bounded_generation"] += (
                key in actual and key not in rank
            )
            counts["truth_in_original_handoff"] += key in actual
            counts["legacy_available"] += inherited_available
            if inherited_available and key not in actual:
                # Mutually exclusive attribution; neither branch chooses predictions using truth.
                in_pool, in_generated = key in pool, key in gen
                category = (
                    "retrieval_pool_only"
                    if in_pool and not in_generated
                    else "generated_only"
                    if in_generated and not in_pool
                    else "both_retrieval_pool_and_generated"
                    if in_pool and in_generated
                    else "outside_selected_pool_and_generated"
                )
                missing[category] += 1
                counts["missing_truth_with_reference_protection"] += protected
                counts["missing_truth_without_reference_protection"] += not protected
                counts["missing_truth_with_retrieval_ranking_length_25"] += (
                    len(rank) == 25
                )
        report[mode] = {
            "molecules": expected,
            **dict(counts),
            "legacy_available_but_absent_handoff_by_disjoint_origin": dict(missing),
            "legacy_available_but_absent_final_by_actual_route": dict(final_missing),
        }
        inputs[mode] = {
            "records_sha256": digest(records),
            "confidence_sha256": digest(conf_file),
            "reference_sha256": digest(reference_file),
            "legacy_metrics_sha256": digest(legacy_file),
            "actual_route_audit_sha256": digest(audit_path),
        }
    value = {
        "diagnostic_only": True,
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "holdout_used": False,
        "source_sha256": digest(__file__),
        "input_sha256": inputs,
        "shared_input_sha256": {
            "handoff": digest(HANDOFF),
            "generated": digest(GENERATED),
            "critic_scores": digest(SCORES),
            "promotion_scores": digest(PAIRS),
            "external_proposals": digest(proposal_path),
            "native_proposals": digest(native_path),
            "final_rankings": digest(LEGACY / "selected_rankings.json"),
        },
        "cohorts": report,
        "scope": "Full3340 exact reconstruction of stored original0062 handoffs. Truth used only for aggregate diagnostics. Legacy covered exactly equals original handoff plus full corrected external proposals, not selected retrieval pool or final output; missing truth is not evidence of a live deployment truncation bug. No gate, model, threshold, prediction, holdout or scientific incumbent changed.",
    }
    write_json(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
