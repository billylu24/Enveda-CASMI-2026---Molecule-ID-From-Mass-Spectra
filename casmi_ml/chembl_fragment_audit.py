"""Aggregate-only diagnosis of a frozen ChEMBL fragment development experiment."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_fragment_pilot import (
    informative_fragments,
    possible_critic_gate,
    relative_supported_proposals,
    score_cache_key,
)
from casmi_ml.chembl_sequence_pilot import reorder_scoreable
from casmi_ml.chemistry import rerank
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
from casmi_ml.research_protocol import ROOT


def run(directory, output):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if (
        protocol["limit"] != 2000
        or not protocol["compare_current_first_fragment"]
        or protocol["candidate_gate"]
        or protocol.get("fingerprint_prior_ranking", False)
        or protocol.get("chemical_prior", False)
        or protocol.get("fragment_gate_only", False)
    ):
        raise ValueError("Audit requires complete relative-fragment/critic experiment")
    sequence_values = {}
    if protocol.get("sequence_ranking"):
        sequence_path = Path(
            "artifacts/research_loop/rounds/0110_chembl_sequence_score_full/scores.json"
        )
        if digest(sequence_path) != protocol["sequence_scores_sha256"]:
            raise ValueError("Sequence audit cache differs from frozen experiment")
        sequence_values = json.loads(sequence_path.read_text())
    fragment_scores = json.loads((directory / "fragment_scores.json").read_text())
    pair_scores = json.loads((directory / "critic_scores.json").read_text())
    proposals = json.loads(
        Path(
            "artifacts/research_loop/rounds/0071_chembl_candidate_slots/proposals.json"
        ).read_text()
    )
    scores = json.loads(SCORES.read_text())
    ordered = {
        r["key"]: combined_order(
            r["candidates"],
            scores.get(r["key"], {}).get("fingerprint", {}),
            (1, 0, 0.5),
        )
        for r in json.loads(GENERATED.read_text())
    }
    pairs = json.loads(
        Path(
            "artifacts/research_loop/rounds/0061_generated_first_critic/scores.json"
        ).read_text()
    )
    variant = f"{'fragment1' if protocol.get('fragment_weight', 0.5) == 1.0 else 'fragment05'}_prefix{protocol['insertion_prefix']}"
    spec = protocol["variants"][variant]
    aggregate = {}
    for mode in ("unknown", "known"):
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        observed = set(
            pd.read_parquet(
                Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                / mode
                / "rows.parquet",
                columns=["inchikey14"],
            ).inchikey14
        )
        counts = Counter()
        reciprocal = 0.0
        top1 = 0
        for row in rows:
            if mode == "known" and not row["known"]:
                continue
            counts["molecules"] += 1
            key, base = row["key"], row["variants"]["baseline"]
            protected = protects_reference(base["ranking"], observed, confidence[key])
            selected = base if protected else row["variants"]["expansion_1"]
            prefix = (
                select_prefix(
                    base["ranking"], observed, confidence[key], "second_unreferenced"
                )
                if protected
                else 2
            )
            generated = ordered[key]
            novel = [k for k in generated if k not in set(selected["ranking"])]
            if confidence[key] < 0.2 and selected["ranking"] and novel:
                prefix = promotion_prefix(
                    selected["ranking"],
                    generated,
                    pairs[key + ":" + selected["ranking"][0] + ":" + novel[0]],
                    prefix,
                    0.05,
                )
            current = insert_generated(selected["ranking"], generated, prefix, 5)
            result = current
            if confidence[key] < 0.5 and current:
                counts["low_confidence_queries"] += 1
                native = [k for k in proposals[key] if k not in set(current)][
                    : protocol["proposal_limit"]
                ]
                true_novel = key in native
                counts["novel_truth_in_native"] += true_novel
                ck = score_cache_key(
                    key, current[0], native, protocol.get("proposal_model_binding")
                )
                if native:
                    critic = pair_scores[ck]
                    shortlist = sorted(native, key=lambda k: (-critic[k], k))
                    if protocol.get("sequence_ranking"):
                        shortlist = reorder_scoreable(
                            shortlist,
                            sequence_values.get(key, {}),
                            protocol["sequence_ratio_weight"],
                        )
                    shortlist = shortlist[: protocol["fragment_limit"]]
                    counts["novel_truth_in_critic_shortlist"] += key in shortlist
                    possible = possible_critic_gate(
                        shortlist, critic, current[0], spec[0]
                    )
                    counts["queries_with_possible_critic_gate"] += possible
                    counts["novel_truth_in_possible_group"] += true_novel and possible
                    if (
                        protocol.get("skip_impossible_critic_gate", False)
                        and not possible
                    ):
                        counts["skipped_impossible_group"] += 1
                    else:
                        fk = score_cache_key(
                            "include_current_first_v1:" + key, current[0], shortlist
                        )
                        if protocol.get("fragment_aggregation") == "mean_normalized":
                            fk = score_cache_key(
                                "mean_normalized_v1:" + key, current[0], shortlist
                            )
                        if protocol.get("fragment_aggregation") == "merged_peak_union":
                            fk = score_cache_key(
                                "merged_peak_union_v1:" + key, current[0], shortlist
                            )
                        if protocol.get("fragment_depth", 2) != 2:
                            fk = score_cache_key(
                                f"depth{protocol['fragment_depth']}:" + fk,
                                current[0],
                                shortlist,
                            )
                        saved = fragment_scores[fk]
                        fragments = saved["scores"]
                        proposed = rerank(
                            shortlist,
                            {},
                            [],
                            spec[3],
                            top_n=len(shortlist),
                            fragment_scores=fragments,
                        )
                        gates = {
                            "budget": not saved["budget_fallback"],
                            "positive_informative": informative_fragments(
                                proposed, fragments
                            ),
                            "relative_fragment": fragments.get(proposed[0], 0.0)
                            > fragments.get(current[0], 0.0),
                            "critic_margin": critic[proposed[0]]
                            > critic[current[0]] + spec[0],
                        }
                        for gate, passed in gates.items():
                            counts["queries_passing_" + gate] += passed
                            counts["novel_truth_groups_passing_" + gate] += (
                                true_novel and passed
                            )
                        if true_novel:
                            counts["novel_truth_positive_fragment"] += (
                                fragments.get(key, 0) > 0
                            )
                            counts["novel_truth_fragment_beats_first"] += fragments.get(
                                key, 0
                            ) > fragments.get(current[0], 0)
                            counts["novel_truth_own_critic_passes"] += (
                                critic[key] > critic[current[0]] + spec[0]
                            )
                            counts["novel_truth_proposed_top3"] += (
                                key in proposed[: spec[2]]
                            )
                            counts["novel_truth_budget_fallback"] += saved[
                                "budget_fallback"
                            ]
                        if all(gates.values()):
                            counts["queries_inserted"] += 1
                            if protocol.get("relative_candidate_gate"):
                                proposed = relative_supported_proposals(
                                    proposed, fragments, current[0]
                                )
                            counts["novel_truth_inserted"] += key in proposed[: spec[2]]
                            result = insert_generated(
                                current, proposed, spec[1], spec[2]
                            )
            rank = result.index(key) + 1 if key in result else 0
            reciprocal += 1 / rank if 1 <= rank <= 25 else 0
            top1 += rank == 1
        expected = report[mode][variant]
        actual = {
            "mrr25": reciprocal / counts["molecules"],
            "top1": top1 / counts["molecules"],
        }
        if any(abs(actual[k] - expected[k]) > 1e-12 for k in actual):
            raise ValueError(
                "Audit reconstruction disagrees with frozen ranking report"
            )
        aggregate[mode] = {"counts": dict(counts), "reconstructed_metrics": actual}
    result = {
        "diagnostic_only": True,
        "cohort": "repeated_development",
        "independent_acceptance": False,
        "experiment": directory.name,
        "protocol_sha256": digest(directory / "protocol.json"),
        "report_sha256": digest(directory / "report.json"),
        "truth_used_only_for_aggregate_diagnosis": True,
        "stage_counts_are_not_disjoint": True,
        "aggregate": aggregate,
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.output), indent=2))


if __name__ == "__main__":
    main()
