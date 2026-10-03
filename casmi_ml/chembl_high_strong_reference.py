"""Protect strongly matched original positions while requiring full first3 fragment support."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_critic_slots import score_cache_key
from casmi_ml.chembl_high_fragment_pilot import HIGH, fragment_structures
from casmi_ml.chembl_high_reference_prefix import reference_supported_prefix
from casmi_ml.chembl_routed_inference import select_high_fragment
from casmi_ml.chembl_tail_controls import validate_tail
from casmi_ml.chemistry import rerank
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.generation_slots import insert_generated
from casmi_ml.mass_candidates import mass_centers
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_protocol import ROOT, freeze

BASE = Path("artifacts/research_loop/rounds/0149_chembl_routed_combination")
CLASSES = Path("artifacts/research_loop/metfrag_polling_classes")
JAR = Path("external/metfrag/MetFragCommandLine-2.6.11.jar")
JAVA = Path("external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java")


def strong_reference_keys(reference, query):
    scores = {}
    for center in mass_centers(query, "charge_aware_union"):
        for key, value in reference.rank(query, center):
            scores[key] = max(scores.get(key, 0), value)
    return {key for key, value in scores.items() if value >= 0.5}, scores


def supported_promotion(
    prior, current, proposed, original_scores, fragments, fallback, observed
):
    if fallback or any(key not in original_scores for key in prior[:3]):
        return current, False
    if fragments.get(proposed[0], 0) <= max(original_scores[k] for k in prior[:3]):
        return current, False
    return insert_generated(
        prior, proposed[:3], reference_supported_prefix(prior, observed), 3
    ), True


def run(output, incumbent, limit=200):
    output, incumbent = Path(output), Path(incumbent)
    if limit not in (200, 2000):
        raise ValueError("Fixed hash200 or full2000 required")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((HIGH / "protocol.json").read_text())
    protocol = json.loads((incumbent / "protocol.json").read_text())
    if protocol["prefix"] != 3 or protocol["limit"] != 2000:
        raise ValueError("Frozen163 source required")
    proposals_path = Path(config["proposal_path"])
    if (
        digest(proposals_path) != config["native_proposals_sha256"]
        or digest(DERIVED) != config["catalog_sha256"]
    ):
        raise ValueError("Frozen external assets differ")
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "incumbent_directory": str(incumbent),
            "reference_similarity_threshold": 0.5,
            "reference_score_dependencies_sha256": {
                p: digest(p)
                for p in (
                    "casmi_ml/ranking.py",
                    "casmi_ml/mass_candidates.py",
                    "casmi_ml/chembl_high_reference_prefix.py",
                )
            },
            "reference_matrix_sha256": {
                mode: digest(
                    Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                    / mode
                    / "spectra.npz"
                )
                for mode in ("unknown", "known")
            },
            "reference_rows_sha256": {
                mode: digest(
                    Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                    / mode
                    / "rows.parquet"
                )
                for mode in ("unknown", "known")
            },
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "incumbent_rankings_sha256": digest(incumbent / "selected_rankings.json"),
            "fragment_cache_sha256": digest(incumbent / "fragment_scores.json"),
            "fragment_protocol_sha256": digest(incumbent / "protocol.json"),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "proposals_sha256": digest(proposals_path),
            "critic_scores_sha256": digest(HIGH / "critic_scores.json"),
            "additional_engine": {
                p: digest(p)
                for p in (
                    "casmi_ml/metfrag_persistent.py",
                    str(
                        CLASSES
                        / "de/ipbhalle/metfraglib/process/CombinedMetFragProcess.class"
                    ),
                    str(CLASSES / "MetFragWorker.class"),
                    str(JAR),
                    str(JAVA),
                )
            },
            "rule": "Freeze163 low/fallback/remove-tail branches. Only existing163 supported high insertion groups: score actual available original first3 SMILES on same merged precise monomer inputs, require new proposed leader fragment strictly greater than all original first3 before moving identical3 slots after max(1,last original position with existing charge-aware reference similarity>=.5 within available first3). Original reference-supported first2/3 stay protected. Otherwise keep163 prefix3. Repeated actual-first score must equal frozen score exactly. No new candidates, critic thresholds or weights.",
            "limit": limit,
            "selector": "First hash256 fragment-representative-20261003 keys",
            "fragment_seconds": 1200,
            "backend": "persistent threads2 polling10ms",
            "fragment_score_reuse": "Exact physical original3 scores from178; no new evidence or scores selected by labels",
            "reused_score_files_sha256": {
                p.name: digest(p)
                for p in sorted((output / "original_fragment_cache").glob("*.json"))
            },
            "new_training": False,
            "new_sampling": False,
            "cohort": "repeated_development",
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "independent_acceptance": False,
        },
    )
    baseline = json.loads((incumbent / "selected_rankings.json").read_text())
    routed = json.loads((BASE / "selected_rankings.json").read_text())
    original = json.loads((BASE / "low_rankings.json").read_text())["baseline"]
    proposals = json.loads(proposals_path.read_text())
    pairs = json.loads((HIGH / "critic_scores.json").read_text())
    cache = json.loads((incumbent / "fragment_scores.json").read_text())
    binding = {k: config[k] for k in ("encoder_sha256", "critic_sha256")}
    external = (
        pd.read_parquet(DERIVED, columns=["inchikey14", "normalized_smiles"])
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    pubchem = pd.read_parquet(
        "external/pubchemlite/structures.parquet",
        columns=["inchikey14", "normalized_smiles"],
    )
    for row in pubchem.itertuples():
        lookup.setdefault(row.inchikey14, row.normalized_smiles)
    del pubchem
    generated = {
        r["key"]: {c["key"]: c["smiles"] for c in r["candidates"]}
        for r in json.loads(GENERATED.read_text())
    }
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    allowed = set(
        sorted(
            groups,
            key=lambda k: hashlib.sha256(
                ("fragment-representative-20261003:" + k).encode()
            ).digest(),
        )[:limit]
    )
    started = time.monotonic()
    engine = PersistentMonomerMetFrag(
        JAR,
        output / "original_fragment_cache",
        java=str(JAVA),
        classes=CLASSES,
        threads=2,
    )
    report, selected, statistics = {}, {}, {}
    try:
        for mode in ("unknown", "known"):
            reference = ReferenceIndex(
                Path("artifacts/research_loop/rounds/0001_mass_v2/reference") / mode
            )
            if mode == "known":
                lookup.update(candidate_lookup(ROOT, "researchdev", "known"))
            confidence = {
                r["key"]: r["confidence"]
                for r in json.loads(
                    (ROOT / f"researchdev_{mode}_chemical.json").read_text()
                )
            }
            base = pd.read_csv(
                incumbent / f"{mode}_high_relative_fragment.csv"
            ).set_index("key")
            selected[mode] = {}
            stats = {
                "original_score_groups": 0,
                "promoted_groups": 0,
                "actual_first_score_exact": 0,
                "budget_fallback": 0,
                "missing_evidence": 0,
            }
            for key, current in baseline[mode].items():
                if key not in allowed:
                    continue
                prior, result = original[mode][key], current
                if confidence[key] >= 0.5 and routed[mode][key] != prior:
                    native = [k for k in proposals[key] if k not in set(prior)][:100]
                    values = pairs[score_cache_key(key, prior[0], native, binding)]
                    candidates = sorted(native, key=lambda k: (-values[k], k))
                    structures = fragment_structures(
                        key, prior[0], candidates, external, lookup, generated
                    )
                    cache_key = score_cache_key(
                        "high_relative_merged_actual_first_v2:" + key,
                        prior[0],
                        candidates,
                        {
                            "structures": structures,
                            "dependencies": protocol["dependencies_sha256"],
                        },
                    )
                    item = cache[cache_key]
                    reconstructed, _, status = select_high_fragment(
                        prior,
                        routed[mode][key],
                        candidates,
                        values,
                        item["scores"],
                        item["budget_fallback"],
                    )
                    if reconstructed != current:
                        raise ValueError("Frozen163 original reconstruction differs")
                    if status == "high_fragment_inserted":
                        original_structures = {
                            k: generated[key].get(k, lookup.get(k)) for k in prior[:3]
                        }
                        if any(v is None for v in original_structures.values()):
                            raise ValueError(
                                "Actual original first3 representation unavailable"
                            )
                        query = frame.iloc[groups[key]].drop(
                            columns=[
                                c
                                for c in (
                                    "inchikey14",
                                    "normalized_smiles",
                                    "molecular_formula",
                                    "fingerprint",
                                )
                                if c in frame
                            ]
                        )
                        observed, _ = strong_reference_keys(reference, query)
                        scores, fallback = score_group_merged(
                            engine,
                            query.to_dict("records"),
                            original_structures,
                            deadline=started + 1200,
                        )
                        stats["original_score_groups"] += 1
                        if fallback:
                            stats["budget_fallback"] += 1
                        elif scores:
                            if scores.get(prior[0], 0) != item["scores"].get(
                                prior[0], 0
                            ):
                                raise ValueError(
                                    "Candidate-pool independent actual-first fragment score differs"
                                )
                            stats["actual_first_score_exact"] += 1
                        else:
                            stats["missing_evidence"] += 1
                        proposed = rerank(
                            candidates,
                            {},
                            [],
                            0.5,
                            top_n=len(candidates),
                            fragment_scores=item["scores"],
                        )
                        result, promoted = supported_promotion(
                            prior,
                            current,
                            proposed,
                            scores,
                            item["scores"],
                            fallback,
                            observed,
                        )
                        stats["promoted_groups"] += promoted
                        if stats["original_score_groups"] % 25 == 0:
                            print(
                                mode,
                                stats,
                                "seconds",
                                time.monotonic() - started,
                                flush=True,
                            )
                validate_tail(prior, current, result, confidence[key], prefix=1)
                selected[mode][key] = result
            statistics[mode] = stats
            report[mode] = {}
            for name, orders in [
                ("baseline", {k: baseline[mode][k] for k in selected[mode]}),
                ("strong_reference_prefix", selected[mode]),
            ]:
                per = base.loc[list(orders)].copy()
                for key, order in orders.items():
                    rank = order.index(key) + 1 if key in order else 0
                    per.loc[key, ["reciprocal_rank", "top1", "top5", "top25"]] = [
                        1 / rank if 0 < rank <= 25 else 0,
                        int(rank == 1),
                        int(0 < rank <= 5),
                        int(0 < rank <= 25),
                    ]
                report[mode][name] = {
                    "molecules": len(per),
                    "candidate_recall": float(per.covered.mean()),
                    "mrr25": float(per.reciprocal_rank.mean()),
                    **{k: float(per[k].mean()) for k in ("top1", "top5", "top25")},
                    "conditional_mrr25": float(
                        per.loc[per.covered == 1, "reciprocal_rank"].mean()
                    ),
                }
                per.reset_index().to_csv(output / f"{mode}_{name}.csv", index=False)
    finally:
        engine.close()
    write_json(output / "selected_rankings.json", selected)
    report.update(
        diagnostic_only=limit != 2000,
        independent_acceptance=False,
        diagnostics={
            "counts": statistics,
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        },
    )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--limit", type=int, choices=(200, 2000), default=200)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.incumbent, a.limit), indent=2))


if __name__ == "__main__":
    main()
