"""Frozen0062 rankings plus bounded ChEMBL proposals on low-confidence queries."""

import argparse
import json
import resource
import time
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_catalog import DERIVED
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
from casmi_ml.inference import group_probability
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex, metrics, neural_rank
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

VARIANTS = {
    "baseline": None,
    "low02_prefix2_slots3": (0.2, 2, 3),
    "low02_prefix5_slots3": (0.2, 5, 3),
    "low05_prefix5_slots3": (0.5, 5, 3),
}


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    pair_path = Path(
        "artifacts/research_loop/rounds/0061_generated_first_critic/scores.json"
    )
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "catalog_sha256": digest(DERIVED),
            "encoder_sha256": digest(ENCODER),
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "promotion_scores_sha256": digest(pair_path),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": VARIANTS,
            "rule": "Freeze0062 complete generated/retrieval rank first; add at most3 novel native-fingerprint-ranked ChEMBL candidates after first2/5 under fixed actual retrieval confidence; existing rank order preserved",
            "mass_hypothesis": "charge_aware_union",
            "new_training": False,
            "new_sampling": False,
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "cohort": "repeated_development",
            "score_seconds": 3600,
        },
    )
    configure(42, threads=4)
    manifest = json.loads(Path("external/chembl37/manifest.json").read_text())
    if digest(DERIVED) != manifest["derived_sha256"]:
        raise ValueError("External source changed")
    index = CandidateIndex(pd.read_parquet(DERIVED))
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    generated = json.loads(GENERATED.read_text())
    scores = json.loads(SCORES.read_text())
    ordered = {
        r["key"]: combined_order(
            r["candidates"],
            scores.get(r["key"], {}).get("fingerprint", {}),
            (1, 0, 0.5),
        )
        for r in generated
    }
    pairs = json.loads(pair_path.read_text())
    if len(generated) != 2000 or set(ordered) != set(groups):
        raise ValueError("Full2000 keys required")
    encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
    encoder.eval()
    cache_path = output / "proposals.json"
    proposals = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    budget = StageBudget(
        output,
        "catalog_cpu",
        "proposal_scoring",
        3600,
        limit=3600,
        lock_path=output / "score.lock",
    )
    started = time.monotonic()
    report, diagnostics = {}, {}
    try:
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
            if {r["key"] for r in rows} != set(groups):
                raise ValueError("Retrieval keys differ")
            ranks = {name: {} for name in VARIANTS}
            pools = {name: {} for name in VARIANTS}
            diagnostics[mode] = {
                name: {
                    "queries_with_inserted_candidates": 0,
                    "novel_inserted_truths": 0,
                }
                for name in VARIANTS
            }
            for i, row in enumerate(rows, 1):
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                protected = protects_reference(
                    base["ranking"], observed, confidence[key]
                )
                selected = base if protected else row["variants"]["expansion_1"]
                prefix = (
                    select_prefix(
                        base["ranking"],
                        observed,
                        confidence[key],
                        "second_unreferenced",
                    )
                    if protected
                    else 2
                )
                candidates = ordered[key]
                novel = [k for k in candidates if k not in set(selected["ranking"])]
                if confidence[key] < 0.2 and selected["ranking"] and novel:
                    prefix = promotion_prefix(
                        selected["ranking"],
                        candidates,
                        pairs[key + ":" + selected["ranking"][0] + ":" + novel[0]],
                        prefix,
                        0.05,
                    )
                current = insert_generated(selected["ranking"], candidates, prefix, 5)
                current_pool = selected["pool"] + candidates
                if confidence[key] < 0.5 and key not in proposals:
                    if not budget.checkpoint():
                        raise TimeoutError("Proposal scoring budget exhausted")
                    query = frame.iloc[groups[key]].drop(
                        columns=[
                            c
                            for c in [
                                "inchikey14",
                                "normalized_smiles",
                                "molecular_formula",
                                "fingerprint",
                            ]
                            if c in frame
                        ]
                    )
                    # Keep mass-window fingerprint caching bounded for deployment.
                    if len(index.cache) > 20000:
                        index.cache.clear()
                    pool, fps = candidate_window(index, query, "charge_aware_union")
                    probability = group_probability(
                        encoder, query, saved["preprocessing"]
                    )
                    proposals[key] = neural_rank(
                        probability, pool.inchikey14.tolist(), fps
                    )
                    if len(proposals) % 25 == 0:
                        write_json(cache_path, proposals)
                        print(
                            "chembl_proposals",
                            len(proposals),
                            "seconds",
                            time.monotonic() - started,
                            flush=True,
                        )
                for name, spec in VARIANTS.items():
                    external = (
                        proposals.get(key, [])
                        if spec and confidence[key] < spec[0]
                        else []
                    )
                    novel_external = (
                        [k for k in external if k not in set(current)][: spec[2]]
                        if spec
                        else []
                    )
                    ranks[name][key] = (
                        insert_generated(current, external, spec[1], spec[2])
                        if external
                        else current
                    )
                    pools[name][key] = current_pool + external
                    diagnostics[mode][name]["queries_with_inserted_candidates"] += bool(
                        novel_external
                    )
                    diagnostics[mode][name]["novel_inserted_truths"] += (
                        key in novel_external
                    )
            report[mode] = {}
            for name in VARIANTS:
                result, per = metrics(ranks[name], pools[name])
                if name == "baseline":
                    expected = (
                        pd.read_csv(incumbent / f"{mode}_confidence0.2_margin0.05.csv")
                        .set_index("key")
                        .sort_index()
                    )
                    actual = per.set_index("key").sort_index()
                    if (
                        not actual.index.equals(expected.index)
                        or (
                            actual[["reciprocal_rank", "top1"]]
                            - expected[["reciprocal_rank", "top1"]]
                        )
                        .abs()
                        .max()
                        .max()
                        > 1e-12
                    ):
                        raise ValueError("Paired baseline must match0062")
                report[mode][name] = result
                per.to_csv(output / f"{mode}_{name}.csv", index=False)
        write_json(cache_path, proposals)
        write_json(output / "diagnostics.json", diagnostics)
        report["diagnostics"] = {
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "proposal_queries_scored": len(proposals),
            "variants": diagnostics,
            "independent_acceptance": False,
        }
        write_json(output / "report.json", report)
        return report
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.incumbent), indent=2))


if __name__ == "__main__":
    main()
