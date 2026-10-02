"""Frozen chemical reranking development and one-use independent acceptance."""

import argparse
import fcntl
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml import ablation
from casmi_ml.chemistry import extract_evidence, rerank, rule_manifest, score_candidate
from casmi_ml.data import cache_features, fingerprint, write_json
from casmi_ml.experiment import bootstrap_difference
from casmi_ml.inference import group_probability
from casmi_ml.metfrag import MetFrag, digest
from casmi_ml.ranking import build_candidates, metrics, neural_rank
from casmi_ml.research_protocol import CATALOG, ENCODER, ROOT, freeze, prepare
from casmi_ml.scale_experiment import COCONUT, SelectedFeatures
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

ROUTING = {"kind": "free_top1", "base": "coconut15", "threshold": 0.5, "weight": 0.75}


@torch.inference_mode()
def _reference_records(root, split, mode):
    root = Path(root)
    probability_path = root / f"{split}_probabilities.npy"
    if not probability_path.exists():
        configure(threads=4)
        model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
        directory = root / "features" / split
        if not (directory / "complete.json").exists():
            cache_features(
                pd.read_parquet(root / f"{split}.parquet"),
                checkpoint["preprocessing"],
                directory,
            )
        dataset = SelectedFeatures(directory, model.input_names, target=False)
        output = [
            model(batch).sigmoid().numpy()
            for batch in torch.utils.data.DataLoader(dataset, batch_size=128)
        ]
        np.save(probability_path, np.concatenate(output))
        del model, dataset, output
        gc.collect()
    # Existing historical/control builder uses the same neural probability cache.
    previous = ablation.ROOT
    ablation.ROOT = root
    try:
        return ablation.build_records(split, mode)
    finally:
        ablation.ROOT = previous


def reference_records(root, split, mode):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "reference_build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _reference_records(root, split, mode)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def candidate_lookup(root, split, mode):
    catalog = pd.read_parquet(CATALOG)
    if mode == "known":
        frame = pd.read_parquet(Path(root) / f"{split}.parquet", columns=["inchikey14"])
        observed = set(
            pd.read_parquet(
                Path(root) / "reference" / f"{split}_known" / "rows.parquet",
                columns=["inchikey14"],
            ).inchikey14
        )
        catalog = catalog[
            ((catalog.split == "train") | catalog.inchikey14.isin(frame.inchikey14))
            & catalog.inchikey14.isin(observed)
        ]
    pool = build_candidates(catalog, COCONUT, final=mode == "known")
    return dict(zip(pool.inchikey14, pool.normalized_smiles))


@torch.inference_mode()
def deployable_records(root, split, mode):
    root = Path(root)
    records = reference_records(root, split, mode)
    if split != "researchholdout":
        return records
    path = root / f"{split}_{mode}_deployment_records.json"
    spec = {
        "encoder_sha256": digest(ENCODER),
        "input_sha256": digest(root / f"{split}.parquet"),
        "reference_records_sha256": digest(root / f"{split}_{mode}_records.json"),
        "neural_inference": "CPU float32 actual group_probability deployment path",
    }
    freeze(path.with_suffix(".config.json"), spec)
    if path.exists():
        return json.loads(path.read_text())
    frame = pd.read_parquet(root / f"{split}.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
    lookup = candidate_lookup(root, split, mode)
    cache = {}
    output = []
    for count, record in enumerate(records, 1):
        keys = record["available"]["union35"]
        for key in keys:
            if key not in cache:
                cache[key] = fingerprint(lookup[key])
        fps = np.asarray([cache[k] for k in keys], dtype=np.float32).reshape(-1, 2048)
        probability = group_probability(
            model, frame.iloc[groups[record["key"]]], checkpoint["preprocessing"]
        )
        neural = neural_rank(probability, keys, fps)
        output.append({**record, "neural": neural})
        if count % 1000 == 0:
            print("deployment records", split, mode, count, flush=True)
    write_json(path, output)
    return output


def _chemical_records(root, split, mode, metfrag=None):
    root = Path(root)
    out = root / f"{split}_{mode}_chemical.json"
    spec = {
        "rules": rule_manifest(),
        "fragmenter_sha256": metfrag.sha256 if metfrag else None,
        "baseline_encoder": digest(ENCODER),
    }
    spec = json.loads(json.dumps(spec))
    protocol_path = root / "chemical_protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        stripped = json.loads(json.dumps(old))
        clean = json.loads(json.dumps(spec))
        for value in [stripped, clean]:
            for rule in value["rules"]["rules"]:
                rule.pop("source", None)
        if stripped == clean and old != spec:
            write_json(
                root / "citation_correction.json",
                {
                    "previous": old,
                    "corrected": spec,
                    "note": "Bibliographic correction only; masses, SMARTS, scores and selection are unchanged.",
                },
            )
            write_json(protocol_path, spec)
    freeze(protocol_path, spec)
    if out.exists():
        return json.loads(out.read_text())
    baseline = deployable_records(root, split, mode)
    lookup = candidate_lookup(root, split, mode)
    frame = pd.read_parquet(root / f"{split}.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    partial = root / f"{split}_{mode}_chemical.partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    completed = {r["key"] for r in rows}

    def compute(record):
        group = frame.iloc[groups[record["key"]]]
        evidences = [extract_evidence(r) for r in group.to_dict("records")]
        base = ablation.route(record, ROUTING)
        historical = record["rankings"]["coconut15"]
        needed = set(base) | set(record["current_full"]) | set(historical)
        # All candidate representations come from reference/public catalogs.
        if not needed <= lookup.keys():
            raise ValueError("Candidate representation missing from scenario pool")
        chemical = {
            component: {
                k: score_candidate(evidences, lookup[k], component)["score"]
                for k in base[:100]
            }
            for component in ["diagnostic", "loss", "combined"]
        }
        fragments, status = {}, []
        if metfrag is not None and record["confidence"] < 0.5:
            candidates = {k: lookup[k] for k in base[:100]}
            fragment_results = [
                metfrag.score(raw, candidates) for raw in group.to_dict("records")
            ]
            for result in fragment_results:
                status.append(result["status"])
                for k, s in result["scores"].items():
                    fragments[k] = max(fragments.get(k, 0.0), s)
        return {
            **record,
            "base": base,
            "chemistry": chemical,
            "fragment_scores": fragments,
            "fragment_status": status,
            "source": "|".join(sorted(set(group.ingest_lib))),
            "query_spectra": len(group),
            "adducts": "|".join(sorted(set(group.adduct))),
            "instrument": "|".join(sorted(str(s) for s in group.instrument_type)),
            "evidence_matches": sum(len(e["matches"]) for e in evidences),
            "shortlist_hit": record["key"] in base[:100],
        }

    # Four independent groups at a time, each with one bounded Java child.
    # Ordered consumption preserves exact output and restart order.
    pending = [r for r in baseline if r["key"] not in completed]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for offset in range(0, len(pending), 16):
            for result in executor.map(compute, pending[offset : offset + 16]):
                rows.append(result)
                if len(rows) % 100 == 0:
                    write_json(partial, rows)
                    print("chemistry", split, mode, len(rows), flush=True)
    write_json(out, rows)
    return rows


def chemical_records(root, split, mode, metfrag=None):
    root = Path(root)
    with (root / f"{split}_{mode}_chemical.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _chemical_records(root, split, mode, metfrag)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def chemical_ranking(row, component, weight):
    if weight == 0 or row["confidence"] >= 0.5:
        return row["base"]
    if component == "combined_fragment":
        # Fuse the two evidence ranks, never add incomparable raw score scales.
        first = rerank(
            row["base"], {}, [], weight, fragment_scores=row["chemistry"]["combined"]
        )
        return rerank(first, {}, [], weight, fragment_scores=row["fragment_scores"])
    scores = (
        row["fragment_scores"]
        if component == "fragment"
        else row["chemistry"][component]
    )
    return rerank(row["base"], {}, [], weight, fragment_scores=scores)


def evaluate(rows, component="combined", weight=0.0, mode="unknown", historical=False):
    chosen = [r for r in rows if mode != "known" or r["known"]]
    rankings = {
        r["key"]: r["rankings"]["coconut15"]
        if historical
        else chemical_ranking(r, component, weight)
        for r in chosen
    }
    pools = {
        r["key"]: list(
            set(r["available"]["union35"]) | set(r["available"]["coconut15"])
        )
        for r in chosen
    }
    report, per = metrics(rankings, pools)
    meta = pd.DataFrame(
        [
            {
                k: r[k]
                for k in [
                    "key",
                    "source",
                    "query_spectra",
                    "adducts",
                    "instrument",
                    "confidence",
                    "shortlist_hit",
                ]
            }
            for r in chosen
        ]
    )
    per = per.merge(meta, on="key", validate="one_to_one")
    report["shortlist_recall100"] = float(per.shortlist_hit.mean())
    report["strata"] = {}
    for dimension in ["source", "adducts", "instrument", "query_spectra", "covered"]:
        report["strata"][dimension] = {
            str(k): {
                "molecules": len(g),
                "mrr25": float(g.reciprocal_rank.mean()),
                "top1": float(g.top1.mean()),
            }
            for k, g in per.groupby(dimension)
        }
    return report, per


def develop(root=ROOT, metfrag=None):
    root = Path(root)
    prepare(root)
    selection = root / "chemical_selection.json"
    if selection.exists():
        return json.loads(selection.read_text())
    rows = {
        m: chemical_records(root, "researchdev", m, metfrag)
        for m in ["unknown", "known"]
    }
    baseline = {m: evaluate(rows[m], mode=m)[0] for m in rows}
    results = []
    components = ["diagnostic", "loss", "combined"] + (
        ["fragment", "combined_fragment"] if metfrag else []
    )
    for component in components:
        for weight in [0.0, 0.25, 0.5]:
            reports = {m: evaluate(rows[m], component, weight, m)[0] for m in rows}
            eligible = (
                reports["unknown"]["mrr25"] > baseline["unknown"]["mrr25"]
                and reports["known"]["mrr25"] >= baseline["known"]["mrr25"] - 0.001
                and reports["known"]["top1"] >= baseline["known"]["top1"] - 0.005
            )
            results.append(
                {
                    "component": component,
                    "weight": weight,
                    "reports": reports,
                    "eligible": eligible,
                }
            )
    candidates = [r for r in results if r["eligible"]]
    winner = (
        max(candidates, key=lambda r: (r["reports"]["unknown"]["mrr25"], -r["weight"]))
        if candidates
        else None
    )
    result = {
        "winner": winner,
        "baseline": baseline,
        "accepted_for_holdout": winner is not None,
        "protocol_sha256": digest(root / "protocol.json"),
        "results": results,
        "historical_control": {
            m: evaluate(rows[m], mode=m, historical=True)[0] for m in rows
        },
    }
    write_json(selection, result)
    for mode in rows:
        evaluate(rows[mode], mode=mode)[1].to_csv(
            root / f"dev_{mode}_baseline.csv", index=False
        )
    return result


def accept(root=ROOT, metfrag=None):
    root = Path(root)
    selection = json.loads((root / "chemical_selection.json").read_text())
    if digest(root / "protocol.json") != selection["protocol_sha256"]:
        raise ValueError("Protocol changed after selection")
    if not selection["accepted_for_holdout"]:
        return {
            "accepted": False,
            "reason": "no_development_winner",
            "holdout_opened": False,
        }
    path = root / "chemical_acceptance.json"
    if path.exists():
        return json.loads(path.read_text())
    winner = selection["winner"]
    reports = {}
    for mode in ["unknown", "known"]:
        rows = chemical_records(root, "researchholdout", mode, metfrag)
        new, a = evaluate(rows, winner["component"], winner["weight"], mode)
        old, b = evaluate(rows, mode=mode)
        a.to_csv(root / f"holdout_{mode}_selected.csv", index=False)
        b.to_csv(root / f"holdout_{mode}_baseline.csv", index=False)
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
        "reports": reports,
        "selection_sha256": digest(root / "chemical_selection.json"),
        "no_retuning": True,
    }
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "develop", "accept"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--metfrag-jar", type=Path)
    args = parser.parse_args()
    configure(threads=4)
    metfrag = (
        MetFrag(args.metfrag_jar, args.root / "metfrag_cache")
        if args.metfrag_jar
        else None
    )
    start = time.monotonic()
    result = (
        prepare(args.root)
        if args.stage == "prepare"
        else develop(args.root, metfrag)
        if args.stage == "develop"
        else accept(args.root, metfrag)
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "seconds": time.monotonic() - start,
                "accepted": result.get("accepted", result.get("accepted_for_holdout")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
