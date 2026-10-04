"""Observe original critic and physical scores without changing any predictions."""

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml import chembl_routed_inference as inference
from casmi_ml import full_gap_candidate_system as system
from casmi_ml.cached_structure_keys import structure_key
from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.generation_sampling import sampling_seed
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze


def run(output, model):
    output = Path(output)
    identity = {}
    inputs = {}
    for mode in ("unknown", "known"):
        path = CONTROL / mode / "test.parquet"
        frame = pd.read_parquet(path)
        inputs[mode] = digest(path)
        for key, group in frame.groupby("molecule_id", sort=False):
            signature = sampling_seed(group.drop(columns=["molecule_id"]))
            if signature in identity and identity[signature] != key:
                raise ValueError("Ambiguous query signature cannot be attributed")
            identity[signature] = key
    freeze(
        output / "trace_protocol.json",
        {
            "source_sha256": digest(__file__),
            "inputs": inputs,
            "base_source_sha256": digest(system.__file__),
            "rule": "Observe original zero-bonus critic arrays and merged physical score dictionaries after their unchanged execution. Require all3340 fullrank exact, no new physical failure or budget fallback. Query truth only enters final aggregate attribution, never scorer/selection arguments. No new training, candidates, gates or holdout.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    records = {}
    active_mode = None
    original_extend = system.extend

    def extend(*args, **kwargs):
        nonlocal active_mode
        active_mode = Path(args[0]).parent.name
        if active_mode not in ("unknown", "known"):
            raise ValueError("Explicit cohort-scoped observation required")
        return original_extend(*args, **kwargs)

    original_loader = system.load_model
    original_fragment = inference.score_group_merged

    def loader(*args, **kwargs):
        score, weights = original_loader(*args, **kwargs)

        def observe(ranker, encoder, group, prep, candidates, fps):
            values = score(ranker, encoder, group, prep, candidates, fps)
            key = identity[sampling_seed(group)]
            keys = [structure_key(s) for s in candidates.normalized_smiles]
            if len(keys) != len(set(keys)):
                raise ValueError("Candidate identity collision in trace")
            records[(active_mode, key)] = {
                "original_first": keys[0],
                "critic": dict(zip(keys, map(float, values))),
                "physical": {},
            }
            return values

        return observe, weights

    def fragments(engine, rows, candidates, **kwargs):
        values, fallback = original_fragment(engine, rows, candidates, **kwargs)
        key = identity[sampling_seed(pd.DataFrame(rows))]
        record = records[(active_mode, key)]
        for k, v in values.items():
            if k in record["physical"] and record["physical"][k] != v:
                raise ValueError("Repeated original physical value changed")
            record["physical"][k] = v
        return values, fallback

    with (
        patch.object(system, "load_model", new=loader),
        patch.object(system, "extend", new=extend),
        patch.object(inference, "score_group_merged", new=fragments),
    ):
        report = system.run(output, model, control=True)
    if report is None:
        return None
    if (
        not report["control_passed"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Exact complete physical control required for attribution")
    final = json.loads((output / "selected_rankings.json").read_text())
    cohorts = {}
    for mode in ("unknown", "known"):
        rows = pd.read_csv(
            system.LOCAL_CONTROL / f"{mode}_supported_leader_replay.csv"
        ).set_index("key")
        counts = Counter()
        for key in rows.index:
            if not rows.loc[key, "covered"] or key in final[mode][key]:
                continue
            counts["legacy_available_absent_final"] += 1
            record = records.get((mode, key))
            if record is None or key not in record["critic"]:
                counts["not_scored_in_actual_shortlist"] += 1
                continue
            counts["scored_truth"] += 1
            critic, physical, first = (
                record["critic"],
                record["physical"],
                record["original_first"],
            )
            counts["truth_passes_individual_critic_margin"] += (
                critic[key] > critic[first] + 0.05
            )
            candidates = sorted(
                (k for k in critic if k != first), key=lambda k: (-critic[k], k)
            )
            for n in (3, 5, 10, 25):
                counts[f"truth_in_critic_first{n}"] += key in candidates[:n]
            if not physical:
                counts["no_physical_dictionary_for_query"] += 1
                continue
            counts["truth_positive_physical_score"] += physical.get(key, 0.0) > 0
            counts["truth_beats_original_first_physical"] += physical.get(
                key, 0.0
            ) > physical.get(first, 0.0)
            counts["truth_passes_both_original_first_gates"] += (
                physical.get(key, 0.0) > physical.get(first, 0.0)
                and critic[key] > critic[first] + 0.05
            )
            fused = rerank(
                candidates, {}, [], 0.5, top_n=len(candidates), fragment_scores=physical
            )
            for n in (3, 5, 10, 25):
                counts[f"truth_in_fragment_fused_first{n}"] += key in fused[:n]
        cohorts[mode] = {"molecules": len(rows), **dict(counts)}
    value = {
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "diagnostic_only": True,
        "control_all3340_full_ranks_exact": True,
        "source_sha256": digest(__file__),
        "control_report_sha256": digest(output / "report.json"),
        "cohorts": cohorts,
        "scope": "Original unchanged scorer/physics traced before aggregate truth diagnosis. Query-level status is not per-candidate rejection reason. Scores are descriptive repeated-development evidence, no source activation or generalization claim.",
    }
    write_json("results/research_loop/candidate_gate_trace.json", value)
    print(json.dumps(value), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
