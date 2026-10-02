"""Development-only representation selection and frozen CPU acceptance/reporting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml import ablation
from casmi_ml.chemistry_experiment import ROUTING
from casmi_ml.data import write_json
from casmi_ml.experiment import bootstrap_difference
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics, neural_rank, rrf
from casmi_ml.representation_experiment import Development, cache, predict
from casmi_ml.research_models import PeakEncoder
from casmi_ml.research_protocol import ROOT, freeze
from casmi_ml.training import configure


def select_representations(root=ROOT):
    root = Path(root)
    path = root / "representation_selection.json"
    if path.exists():
        return json.loads(path.read_text())
    results = []
    for method in ["supervised", "masked", "dino"]:
        run = root / "representation" / f"{method}_42_v2"
        result_path = run / "result.json"
        if not result_path.exists():
            raise ValueError(f"Missing fair comparison {method}")
        result = json.loads(result_path.read_text())
        checkpoint = run / "model.pt"
        item = {
            "method": method,
            "status": result["status"],
            "eligible": result["accepted_for_holdout"],
        }
        if checkpoint.exists():
            c = torch.load(checkpoint, map_location="cpu", weights_only=True)
            item.update(
                {
                    "checkpoint": str(checkpoint),
                    "sha256": digest(checkpoint),
                    "report": c["report"],
                    "baseline": c["baseline"],
                    "epochs": result["best_epoch"],
                }
            )
        results.append(item)
    eligible = [r for r in results if r["eligible"] and "report" in r]
    winner = (
        max(eligible, key=lambda r: r["report"]["routed"]["unknown"]["mrr25"])
        if eligible
        else None
    )
    selection = {
        "winner": winner,
        "results": results,
        "accepted_for_holdout": winner is not None,
        "dev_sha256": digest(root / "researchdev.parquet"),
        "frozen_before_representation_holdout": True,
    }
    freeze(path, selection)
    return selection


def representation_accept(root=ROOT):
    root = Path(root)
    configure(threads=4)
    selection = json.loads((root / "representation_selection.json").read_text())
    if not selection["accepted_for_holdout"]:
        return {
            "accepted": False,
            "holdout_opened": False,
            "reason": "no_development_winner",
        }
    path = root / "representation_acceptance.json"
    if path.exists():
        return json.loads(path.read_text())
    winner = selection["winner"]
    if digest(winner["checkpoint"]) != winner["sha256"]:
        raise ValueError("Selected representation checkpoint changed")
    checkpoint = torch.load(winner["checkpoint"], map_location="cpu", weights_only=True)
    model = PeakEncoder(checkpoint["metadata_dim"]).eval()
    model.load_state_dict(checkpoint["state_dict"])
    evaluation = Development(root, "researchholdout")
    probabilities, _ = predict(
        model, cache(root, "researchholdout"), torch.device("cpu")
    )
    reports = {}
    from casmi_ml.data import fingerprint

    for mode, records in evaluation.records.items():
        rankings, baselines, pools = {}, {}, {}
        for r in records:
            if mode == "known" and not r["known"]:
                continue
            keys = r["available"]["union35"]
            fps = np.array(
                [fingerprint(evaluation.lookup[mode][k]) for k in keys],
                dtype=np.float32,
            ).reshape(-1, 2048)
            neural = neural_rank(
                probabilities[evaluation.groups[r["key"]]].mean(0), keys, fps
            )
            rankings[r["key"]] = ablation.route({**r, "neural": neural}, ROUTING)
            baselines[r["key"]] = ablation.route(r, ROUTING)
            pools[r["key"]] = list(set(keys) | set(r["available"]["coconut15"]))
        new, a = metrics(rankings, pools)
        old, b = metrics(baselines, pools)
        a.to_csv(root / f"representation_holdout_{mode}_selected.csv", index=False)
        b.to_csv(root / f"representation_holdout_{mode}_baseline.csv", index=False)
        reports[mode] = {
            "selected": new,
            "baseline": old,
            "paired": bootstrap_difference(a, b),
        }
    accepted = (
        reports["unknown"]["paired"]["ci95"][0] > 0
        and reports["known"]["selected"]["mrr25"]
        >= reports["known"]["baseline"]["mrr25"] - 0.001
        and reports["known"]["selected"]["top1"]
        >= reports["known"]["baseline"]["top1"] - 0.005
    )
    result = {
        "accepted": bool(accepted),
        "holdout_opened": True,
        "device": "cpu_float32",
        "reports": reports,
        "selection_sha256": digest(root / "representation_selection.json"),
        "no_retuning": True,
    }
    write_json(path, result)
    return result


def generated_report(root, generated, split="researchdev"):
    """Exact structure results, including pure generation and protected retrieval merge."""
    root = Path(root)
    frame = pd.read_parquet(root / f"{split}.parquet")
    known_keys = set(
        pd.read_parquet(
            "artifacts/scale_20260929/train60k.parquet", columns=["inchikey14"]
        ).inchikey14
    )
    from casmi_ml.chemistry_experiment import reference_records

    records = {r["key"]: r for r in reference_records(root, split, "unknown")}
    pure, merged, baseline, pools, baseline_pools = {}, {}, {}, {}, {}
    stats = {
        "samples": 0,
        "terminated": 0,
        "valid": 0,
        "mass_matching": 0,
        "unique_mass_matching": 0,
    }
    formula_hits = 0
    tanimoto_max, tanimoto_first = [], []
    new_truth_hits = 0
    for row in generated:
        key = row["key"]
        r = records[key]
        rank = [c["key"] for c in row["candidates"]]
        base = ablation.route(r, ROUTING)
        pure[key] = rank
        merged[key] = (
            base if r["confidence"] >= 0.5 else rrf([base, rank], [0.75, 0.25])
        )
        baseline[key] = base
        baseline_pools[key] = list(
            set(r["available"]["union35"]) | set(r["available"]["coconut15"])
        )
        pools[key] = list(
            set(r["available"]["union35"])
            | set(r["available"]["coconut15"])
            | set(rank)
        )
        for field in stats:
            stats[field] += row["statistics"][field]
        truth_formula = frame[frame.inchikey14 == key].molecular_formula.iloc[0]
        formula_hits += truth_formula in {
            h["formula"] for h in row["formula_hypotheses"]
        }
        from casmi_ml.data import fingerprint

        truth_smiles = frame[frame.inchikey14 == key].normalized_smiles.iloc[0]
        truth_fp = fingerprint(truth_smiles)
        similarities = []
        for candidate in row["candidates"][:25]:
            fp = fingerprint(candidate["smiles"])
            union = float(np.maximum(truth_fp, fp).sum())
            similarities.append(float(np.minimum(truth_fp, fp).sum()) / max(union, 1.0))
        tanimoto_max.append(max(similarities, default=0.0))
        tanimoto_first.append(similarities[0] if similarities else 0.0)
        new_truth_hits += key in rank and key not in set(
            r["available"]["union35"]
        ) | set(r["available"]["coconut15"])
    pure_report, _ = metrics(pure, {k: pure[k] for k in pure})
    merged_report, merged_rows = metrics(merged, pools)
    old_report, baseline_rows = metrics(baseline, baseline_pools)
    known_records = {r["key"]: r for r in reference_records(root, split, "known")}
    known_new, known_base, known_pools, known_base_pools = {}, {}, {}, {}
    for row in generated:
        key = row["key"]
        r = known_records[key]
        if not r["known"]:
            continue
        rank = [c["key"] for c in row["candidates"]]
        base = ablation.route(r, ROUTING)
        known_base[key] = base
        known_new[key] = (
            base if r["confidence"] >= 0.5 else rrf([base, rank], [0.75, 0.25])
        )
        pool = list(set(r["available"]["union35"]) | set(r["available"]["coconut15"]))
        known_base_pools[key] = pool
        known_pools[key] = list(set(pool) | set(rank))
    known_report = None
    if known_base:
        new, a = metrics(known_new, known_pools)
        old, b = metrics(known_base, known_base_pools)
        known_report = {
            "merged": new,
            "baseline": old,
            "paired": bootstrap_difference(a, b),
        }
    return {
        "molecules": len(generated),
        "known": known_report,
        "pure": pure_report,
        "merged": merged_report,
        "baseline": old_report,
        "merged_paired": bootstrap_difference(merged_rows, baseline_rows),
        "formula_top5": formula_hits / max(len(generated), 1),
        "new_exact_truths_outside_retrieval_pool": int(new_truth_hits),
        "mean_best_top25_tanimoto": float(np.mean(tanimoto_max)),
        "mean_top1_tanimoto": float(np.mean(tanimoto_first)),
        "generated_structure_novel_fraction": sum(
            k not in known_keys
            for r in generated
            for k in [c["key"] for c in r["candidates"]]
        )
        / max(stats["unique_mass_matching"], 1),
        "statistics": stats,
        "valid_rate": stats["valid"] / max(stats["samples"], 1),
        "mass_match_rate": stats["mass_matching"] / max(stats["samples"], 1),
        "unique_rate_among_mass_matches": stats["unique_mass_matching"]
        / max(stats["mass_matching"], 1),
        "labels_used_only_in_metrics": not any(
            r.get("oracle_formula", False) for r in generated
        ),
        "oracle_formula_used": any(r.get("oracle_formula", False) for r in generated),
        "independent_acceptance": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "accept", "generation-report"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--generated", type=Path)
    p.add_argument(
        "--split", default="researchdev", choices=["researchdev", "researchholdout"]
    )
    a = p.parse_args()
    if a.stage == "select":
        result = select_representations(a.root)
    elif a.stage == "accept":
        result = representation_accept(a.root)
    else:
        if not a.generated:
            p.error("--generated required")
        result = generated_report(a.root, json.loads(a.generated.read_text()), a.split)
        write_json(a.generated.with_suffix(".report.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
