"""Aggregate original fixed128 samples: formula, mass validity, and graph recall."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.chemistry import composition_mass, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.full_gap_candidate_system import LOCAL_CONTROL
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.generation_experiment import formula_counts
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT, freeze


def inside_mass(formula, centers):
    value = composition_mass(formula)
    return value is not None and any(
        abs(value - center) <= max(center * 35e-6, 0.006) for center in centers
    )


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Local-only generation funnel required")
    output = Path(output)
    config_path = GENERATED.with_suffix(".config.json")
    config = json.loads(config_path.read_text())
    if (
        config["samples"] != 128
        or config["formula_oracle"]
        or config["limit"] is not None
        or config["sampling"] != "spectrum_hash_v1_and_shared_group_forward_v2"
        or config["split_sha256"] != digest(ROOT / "researchdev.parquet")
    ):
        raise ValueError("Original complete nonoracle fixed128 sampling required")
    generated = json.loads(GENERATED.read_text())
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True)
    selected = json.loads((LOCAL_CONTROL / "selected_rankings.json").read_text())[
        "unknown"
    ]
    if (
        len(generated) != 2000
        or len({r["key"] for r in generated}) != 2000
        or set(groups.groups) != {r["key"] for r in generated}
    ):
        raise ValueError("Complete2000 original generated cohort required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "generated_sha256": digest(GENERATED),
            "generated_config_sha256": digest(config_path),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "final_rankings_sha256": digest(LOCAL_CONTROL / "selected_rankings.json"),
            "rule": "Read original complete2000 frozen nonoracle128samples results only, no "
            "generation/scoring/selection rerun. Aggregate sample validity/mass matching; "
            "recorded top5 predicted formula inside original median neutral mass tolerance; "
            "truth formula supported/in top5, any retained candidate with truth formula, "
            "any exact truth graph, final236 retainedtruth. Alltruth only finalmetrics. "
            "No causal counterfactual/formula-oracle inference, reweighting or new activation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    totals, queries = Counter(), Counter()
    for row in generated:
        if c.stopped():
            return None
        if row["oracle_formula"] or row["statistics"]["samples"] != 128:
            raise ValueError("Original nonoracle sample count required")
        group = groups.get_group(row["key"])
        # Match the original generation mass validation: median, NOT union.
        masses = [
            m
            for r in group.drop(
                columns=["inchikey14", "normalized_smiles", "molecular_formula"],
                errors="ignore",
            ).to_dict("records")
            if (m := neutral_mass(r)) is not None
        ]
        centers = [float(np.median(masses))] if masses else []
        for k, value in row["statistics"].items():
            totals[k] += int(value)
        truth = group.molecular_formula.iloc[0]
        predicted = [r["formula"] for r in row["formula_hypotheses"]]
        candidates = row["candidates"]
        formula_hit = truth in predicted
        graph_hit = any(r["key"] == row["key"] for r in candidates)
        candidate_formula = any(r["formula"] == truth for r in candidates)
        queries["molecules"] += 1
        queries["truth_formula_supported_elements"] += formula_counts(truth) is not None
        queries["predicted_formula_top5_hits"] += formula_hit
        queries["queries_with_any_predicted_top5_formula_mass_compatible"] += any(
            inside_mass(f, centers) for f in predicted
        )
        queries["predicted_top5_formulas_mass_compatible"] += sum(
            inside_mass(f, centers) for f in predicted
        )
        queries["queries_with_valid_generated_samples"] += (
            row["statistics"]["valid"] > 0
        )
        queries["queries_with_retained_mass_matching_candidates"] += bool(candidates)
        queries["queries_with_retained_true_formula_candidate"] += candidate_formula
        queries["queries_with_retained_exact_graph"] += graph_hit
        queries["true_formula_candidate_but_no_exact_graph"] += (
            candidate_formula and not graph_hit
        )
        queries["predicted_formula_hit_but_no_true_formula_candidate"] += (
            formula_hit and not candidate_formula
        )
        queries["exact_graph_without_predicted_formula_top5_hit"] += (
            graph_hit and not formula_hit
        )
        queries["generated_exact_graph_present_in_final236_top25"] += (
            graph_hit and row["key"] in selected[row["key"]][:25]
        )
        queries["generated_exact_graph_absent_final236"] += (
            graph_hit and row["key"] not in selected[row["key"]][:25]
        )
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "sample_totals": dict(totals),
        "query_counts": dict(queries),
        "valid_rate": totals["valid"] / totals["samples"],
        "mass_match_rate": totals["mass_matching"] / totals["samples"],
        "formula_oracle_used": False,
        "predictions_unchanged": True,
        "scope": "Repeated-development frozen original128sample observational funnel. "
        "Predicted formula top5 is not hard decoder conditioning or constraint; "
        "mass-compatible formula search is only diagnosis here. Formula and graph "
        "co-occurrence is not causal or a new accuracy acceptance. No tuning/activation.",
    }
    write_json(output / "report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
