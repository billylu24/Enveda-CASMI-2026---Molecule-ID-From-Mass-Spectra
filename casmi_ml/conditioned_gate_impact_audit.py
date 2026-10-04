"""Observe conditioned score changes through unchanged completed-system gates."""

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml import chembl_routed_inference as inference
from casmi_ml import energy_conditioned_system as conditioned
from casmi_ml import full_gap_candidate_system as system
from casmi_ml.cached_structure_keys import structure_key
from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.direct_models import score_group as original_score
from casmi_ml.generation_sampling import sampling_seed
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

EXPECTED = Path(
    "artifacts/research_loop/rounds/0390_energy_conditioned_chemical_system"
)
MODEL = Path(
    "artifacts/research_loop/rounds/0388_energy_conditioned_chemical/model.json"
)


def run(output):
    output = Path(output)
    original = json.loads((EXPECTED / "report.json").read_text())
    if (
        original["development_budget_fallbacks"]
        or original["incremental_new_failure_contexts"]
    ):
        raise ValueError("Completed conditional system required")
    freeze(
        output / "gate_impact_protocol.json",
        {
            "source_sha256": digest(__file__),
            "system_sha256": digest(conditioned.__file__),
            "model_sha256": digest(MODEL),
            "expected_report_sha256": digest(EXPECTED / "report.json"),
            "expected_rankings_sha256": digest(EXPECTED / "selected_rankings.json"),
            "rule": "Repeat exact390 all2000/1340 full ranks with unchanged conditional model/critic/windows/physics/gates. Observe chemical candidate scores, separately original critic same inputs; no alternate scorer passed into selection. Observe unchanged physical dictionaries. Query truth enters ONLY final aggregate attribution, no input filtering, no selection/gate changes. Diagnose individual critic-margin crossings and frozen fused top3 placement; query-level status alone is not rejection causality. No new model/training/deployment or acceptance.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    identities = {}
    for mode in ("unknown", "known"):
        for key, group in pd.read_parquet(CONTROL / mode / "test.parquet").groupby(
            "molecule_id", sort=False
        ):
            seed = sampling_seed(group.drop(columns="molecule_id"))
            identities[(mode, seed)] = key
    records = {}
    active = None
    loader, fragment, extend = (
        conditioned.load_model,
        inference.score_group_merged,
        system.extend,
    )

    def observed_extend(*args, **kwargs):
        nonlocal active
        active = Path(args[0]).parent.name
        if active not in ("unknown", "known"):
            raise ValueError("Explicit original cohort required")
        return extend(*args, **kwargs)

    def observed_loader(*args, **kwargs):
        score, model = loader(*args, **kwargs)

        def observe(ranker, encoder, group, prep, candidates, fps):
            adjusted = score(ranker, encoder, group, prep, candidates, fps)
            baseline = original_score(ranker, encoder, group, prep, candidates, fps)
            key = identities[(active, sampling_seed(group))]
            keys = [structure_key(s) for s in candidates.normalized_smiles]
            if len(keys) != len(set(keys)):
                raise ValueError("Unchanged unique candidate identity required")
            records[(active, key)] = {
                "first": keys[0],
                "baseline": dict(zip(keys, map(float, baseline))),
                "adjusted": dict(zip(keys, map(float, adjusted))),
                "physical": {},
            }
            return adjusted

        return observe, model

    def observed_fragment(engine, rows, candidates, **kwargs):
        values, fallback = fragment(engine, rows, candidates, **kwargs)
        key = identities[(active, sampling_seed(pd.DataFrame(rows)))]
        for cid, value in values.items():
            prior = records[(active, key)]["physical"].get(cid)
            if prior is not None and prior != value:
                raise ValueError("Same physical context/candidate value changed")
            records[(active, key)]["physical"][cid] = value
        return values, fallback

    with (
        patch.object(conditioned, "load_model", new=observed_loader),
        patch.object(system, "extend", new=observed_extend),
        patch.object(inference, "score_group_merged", new=observed_fragment),
    ):
        report = conditioned.run(output, MODEL)
    if (
        report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete physical repeat required")
    final = json.loads((output / "selected_rankings.json").read_text())
    expected = json.loads((EXPECTED / "selected_rankings.json").read_text())
    if final != expected:
        raise ValueError("All3340 conditional390 full rankings must remain exact")
    write_json(
        output / "private_score_observations.json",
        [
            {"mode": mode, "query": key, **value}
            for (mode, key), value in records.items()
        ],
    )
    cohorts = {}
    for mode in ("unknown", "known"):
        counts = Counter()
        audit = pd.read_csv(output / mode / "selected.csv.external.csv").set_index(
            "molecule_id"
        )
        original_final = json.loads(
            (system.LOCAL_CONTROL / "selected_rankings.json").read_text()
        )[mode]
        for key in final[mode]:
            record = records.get((mode, key))
            if record is None:
                counts["queries_without_scored_shortlist"] += 1
                continue
            counts["queries_scored"] += 1
            first, before, after, physical = (
                record[k] for k in ("first", "baseline", "adjusted", "physical")
            )
            candidates = [cid for cid in after if cid != first]
            old = sorted(candidates, key=lambda cid: (-before[cid], cid))
            new = sorted(candidates, key=lambda cid: (-after[cid], cid))
            counts["queries_critic_candidate_order_changed"] += old != new
            counts["queries_critic_top3_changed"] += old[:3] != new[:3]
            counts["queries_final_fullrank_changed"] += (
                final[mode][key] != original_final[key]
            )
            counts["individual_candidates_crossed_margin_up"] += sum(
                before[cid] <= before[first] + 0.05 and after[cid] > after[first] + 0.05
                for cid in candidates
            )
            counts["individual_candidates_crossed_margin_down"] += sum(
                before[cid] > before[first] + 0.05 and after[cid] <= after[first] + 0.05
                for cid in candidates
            )
            if key not in after:
                counts["truth_not_in_actual_scored_list"] += 1
                continue
            counts["truth_scored"] += 1
            if key == first:
                counts["truth_is_original_first"] += 1
                continue
            counts["truth_critic_margin_crossed_up"] += (
                before[key] <= before[first] + 0.05 and after[key] > after[first] + 0.05
            )
            counts["truth_critic_margin_crossed_down"] += (
                before[key] > before[first] + 0.05 and after[key] <= after[first] + 0.05
            )
            for n in (3, 5, 25):
                counts[f"truth_in_original_critic_top{n}"] += key in old[:n]
                counts[f"truth_in_conditional_critic_top{n}"] += key in new[:n]
            if physical:
                qualified = after[key] > after[first] + 0.05 and physical.get(
                    key, 0
                ) > physical.get(first, 0)
                fused = rerank(
                    new, {}, [], 0.5, top_n=len(new), fragment_scores=physical
                )
                counts["truth_passes_individual_first_critic_and_physical"] += qualified
                counts["qualified_truth_outside_fused_top3"] += (
                    qualified and key not in fused[:3]
                )
                counts["qualified_truth_absent_final"] += (
                    qualified and key not in final[mode][key]
                )
                counts["qualified_absent_truth_high_confidence_queries"] += (
                    qualified
                    and key not in final[mode][key]
                    and audit.loc[key, "confidence"] >= 0.5
                )
        cohorts[mode] = {
            "molecules": len(final[mode]),
            **{k: int(v) for k, v in counts.items()},
        }
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "all3340_conditional390_fullrank_exact": True,
        "development_budget_fallbacks": 0,
        "incremental_new_failure_contexts": 0,
        "cohorts": cohorts,
        "scope": "Aggregate observation through unchanged complete conditional390 system. Individual first critic/physical qualification is descriptive and not sufficient for strong-prefix gates; query status is not a candidate-specific rejection reason. Does not justify relaxing gates or truth-driven candidate selection.",
    }
    write_json(output / "gate_impact_summary.json", result)
    write_json("results/research_loop/conditioned_gate_impact.json", result)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
