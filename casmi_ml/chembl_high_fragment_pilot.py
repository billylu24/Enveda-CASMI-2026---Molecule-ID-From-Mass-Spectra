"""Additional relative fragment evidence for existing0149 high-tail insertions."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_critic_slots import score_cache_key
from casmi_ml.chembl_fragment_pilot import informative_fragments
from casmi_ml.chemistry import rerank
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.generation_slots import insert_generated
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_monomer import MonomerMetFrag
from casmi_ml.research_protocol import ROOT, freeze

HIGH = Path("artifacts/research_loop/rounds/0148_chembl_high_confidence_tail")


def fragment_structures(key, first, candidates, external, original, generated):
    """Use the same actual first SMILES as the frozen original critic."""
    actual_first = generated[key].get(first, original.get(first))
    if actual_first is None:
        raise ValueError("Actual original first representation missing")
    structures = {k: external[k] for k in candidates}
    structures[first] = actual_first
    return structures


def run(output, incumbent, limit=200):
    output, incumbent = Path(output), Path(incumbent)
    if limit not in (200, 2000):
        raise ValueError("Fixed200 or full2000 required")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((HIGH / "protocol.json").read_text())
    dependency_paths = [
        "casmi_ml/metfrag.py",
        "casmi_ml/metfrag_monomer.py",
        "casmi_ml/merged_fragments.py",
        "casmi_ml/chemistry.py",
        "external/metfrag/MetFragCommandLine-2.6.11.jar",
        "external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
    ]
    dependencies = {path: digest(path) for path in dependency_paths}
    selected_path = incumbent / "selected_rankings.json"
    base_path = incumbent / "low_rankings.json"
    for name, path, sha in [
        ("catalog", DERIVED, config["catalog_sha256"]),
        ("proposals", Path(config["proposal_path"]), config["native_proposals_sha256"]),
    ]:
        if digest(path) != sha:
            raise ValueError(f"Frozen {name} differs")
    freeze(
        output / "protocol.json",
        {
            "version": 2,
            "source_sha256": digest(Path(__file__)),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "selected_rankings_sha256": digest(selected_path),
            "original0062_rankings_sha256": digest(base_path),
            "high_protocol_sha256": digest(HIGH / "protocol.json"),
            "high_scores_sha256": digest(HIGH / "critic_scores.json"),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "catalog_sha256": digest(DERIVED),
            "critic_sha256": config["critic_sha256"],
            "encoder_sha256": config["encoder_sha256"],
            "dependencies_sha256": dependencies,
            "first_representation": "Per-query generated SMILES if present, otherwise original candidate lookup with PubChemLite setdefault; identical to original high critic scoring. External representation never overwrites actual first.",
            "limit": limit,
            "pilot_selector": "hash256 fragment-representative-20261003:",
            "rule": "Freeze0149 low and all high noninsertions; only original high critic+.05 passed groups are rescored with merged exact monomer+actual first. Informative complete scores: relative first supported+critic+.05 uses fragment.5 top3 after10; otherwise remove existing tail additions. Missing/noninformative/budget keeps0149 tail unchanged.",
            "prefix": 10,
            "slots": 3,
            "fragment_weight": 0.5,
            "fragment_seconds": 1200,
            "candidate_pool": "Frozen0149 considered pool; all replacements drawn from already included high shortlist100",
            "cohort": "repeated_development",
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "independent_acceptance": False,
        },
    )
    baseline = json.loads(selected_path.read_text())
    original = json.loads(base_path.read_text())["baseline"]
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    keys = sorted(
        groups,
        key=lambda k: hashlib.sha256(
            ("fragment-representative-20261003:" + k).encode()
        ).digest(),
    )[:limit]
    allowed = set(keys)
    proposals = json.loads(Path(config["proposal_path"]).read_text())
    pairs = json.loads((HIGH / "critic_scores.json").read_text())
    binding = {k: config[k] for k in ["encoder_sha256", "critic_sha256"]}
    catalog = pd.read_parquet(DERIVED, columns=["inchikey14", "normalized_smiles"])
    lookup = catalog.set_index("inchikey14").normalized_smiles.to_dict()
    del catalog
    original_lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    pubchem = pd.read_parquet(
        "external/pubchemlite/structures.parquet",
        columns=["inchikey14", "normalized_smiles"],
    )
    original_lookup.update(
        {
            r.inchikey14: r.normalized_smiles
            for r in pubchem.itertuples()
            if r.inchikey14 not in original_lookup
        }
    )
    del pubchem
    generated_lookup = {
        r["key"]: {c["key"]: c["smiles"] for c in r["candidates"]}
        for r in json.loads(GENERATED.read_text())
    }
    fragmenter = MonomerMetFrag(
        "external/metfrag/MetFragCommandLine-2.6.11.jar",
        ROOT / "chembl_metfrag_cache",
        java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
    )
    cache_path = output / "fragment_scores.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    started = time.monotonic()
    report = {}
    counts = {}
    for mode in ("unknown", "known"):
        if mode == "known":
            original_lookup.update(candidate_lookup(ROOT, "researchdev", "known"))
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        baseline_per = pd.read_csv(
            incumbent / f"{mode}_low_fragment_high_tail.csv"
        ).set_index("key")
        selected = {}
        counts[mode] = {
            k: 0
            for k in (
                "eligible_high_groups",
                "informative_groups",
                "tail_removed",
                "tail_reranked",
                "missing_evidence_fallback",
                "budget_fallback",
            )
        }
        for key, current in baseline[mode].items():
            if key not in allowed:
                continue
            result = current
            prior = original[mode][key]
            if confidence[key] >= 0.5 and current != prior:
                counts[mode]["eligible_high_groups"] += 1
                native = [k for k in proposals[key] if k not in set(prior)][:100]
                values = pairs[score_cache_key(key, prior[0], native, binding)]
                candidates = sorted(native, key=lambda k: (-values[k], k))
                structures = fragment_structures(
                    key, prior[0], candidates, lookup, original_lookup, generated_lookup
                )
                scorekey = score_cache_key(
                    "high_relative_merged_actual_first_v2:" + key,
                    prior[0],
                    candidates,
                    {"structures": structures, "dependencies": dependencies},
                )
                if scorekey not in cache:
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
                    scores, fallback = score_group_merged(
                        fragmenter,
                        query.to_dict("records"),
                        structures,
                        deadline=started + 1200,
                    )
                    cache[scorekey] = {"scores": scores, "budget_fallback": fallback}
                    write_json(cache_path, cache)
                    if len(cache) % 10 == 0:
                        print(
                            "high_relative_groups",
                            len(cache),
                            "seconds",
                            time.monotonic() - started,
                            flush=True,
                        )
                scores = cache[scorekey]["scores"]
                proposed = rerank(
                    candidates,
                    {},
                    [],
                    0.5,
                    top_n=len(candidates),
                    fragment_scores=scores,
                )
                if cache[scorekey]["budget_fallback"]:
                    counts[mode]["budget_fallback"] += 1
                elif informative_fragments(proposed, scores):
                    counts[mode]["informative_groups"] += 1
                    if (
                        scores.get(proposed[0], 0) > scores.get(prior[0], 0)
                        and values[proposed[0]] > values[prior[0]] + 0.05
                    ):
                        result = insert_generated(prior, proposed, 10, 3)
                        counts[mode]["tail_reranked"] += 1
                    else:
                        result = prior
                        counts[mode]["tail_removed"] += 1
                else:
                    counts[mode]["missing_evidence_fallback"] += 1
            selected[key] = result
        report[mode] = {}
        for name, ranking in [
            ("baseline", {k: baseline[mode][k] for k in selected}),
            ("high_relative_fragment", selected),
        ]:
            per = baseline_per.loc[list(ranking)].copy()
            for key, order in ranking.items():
                rank = order.index(key) + 1 if key in order else 0
                per.loc[key, ["reciprocal_rank", "top1", "top5", "top25"]] = [
                    1 / rank if 1 <= rank <= 25 else 0,
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
                )
                if per.covered.any()
                else None,
            }
            per.reset_index().to_csv(output / f"{mode}_{name}.csv", index=False)
    report.update(
        diagnostic_only=limit != 2000,
        diagnostics={
            "counts": counts,
            "fragment_groups": len(cache),
            "incremental_spectrum_execution_status_counts": getattr(
                fragmenter, "status_counts", {}
            ),
            "execution_status_scope": "Traversed engine spectra only; complete may be content cache reuse. Incremental elapsed time is not a cold deployment guarantee.",
            "dependencies_sha256": dependencies,
            "actual_first_representation_preserved": True,
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "independent_acceptance": False,
        },
    )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.incumbent, a.limit), indent=2))


if __name__ == "__main__":
    main()
