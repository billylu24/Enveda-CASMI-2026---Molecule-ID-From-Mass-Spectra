"""Single-query, mass-hard-negative contrastive ranking experiment."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml import ablation
from casmi_ml.ablation import sha
from casmi_ml.data import cache_features, fingerprint, write_json
from casmi_ml.direct_models import DirectRanker, batch_graphs, graph
from casmi_ml.experiment import bootstrap_difference
from casmi_ml.ranking import build_candidates, metrics, neural_rank, rrf
from casmi_ml.scale_experiment import (
    COCONUT,
    OLD,
    SelectedFeatures,
    sample_keys,
)
from casmi_ml.scale_experiment import (
    ROOT as SCALE,
)
from casmi_ml.secondary_inference import load_deployment_checkpoint, low_confidence_rank
from casmi_ml.training import configure

ROOT = Path("artifacts/direct_rank_20260929")
ENCODER = SCALE / "runs/wide_enhanced_60000/model.pt"


def protocol():
    spec = {
        "version": 1,
        "encoder_sha256": sha(ENCODER),
        "encoder_frozen": True,
        "representations": ["fingerprint", "graph"],
        "graph": "3 message passing layers, 128 hidden + fingerprint residual branch",
        "training": "60K existing train-only molecules; 15 nearest-mass negatives per example; CE listwise target; random one spectrum per molecule per epoch",
        "negative_pool": "up to 128 nearest-mass training molecules, prioritize within35ppm/floor.006; no evaluation structures as training targets/negatives",
        "evaluation": "one deterministic query spectrum per molecule in both known and unknown scenarios; references rebuilt around single-query masses; known reference excludes query and exact spectrum duplicates",
        "epochs": 16,
        "patience": 4,
        "minimum_epochs": 6,
        "lr": 0.0003,
        "batch_size": 128,
        "neural_blend_weights": [0.25, 0.5, 1.0],
        "routing": {"threshold": 0.5, "weight": 0.75},
        "selection": "max dev unknown routed MRR; known MRR >= baseline-.001 and known top1 >= baseline-.005; require strict unknown improvement",
        "acceptance": "new2000 unknown paired CI95 lower>0; known MRR point delta>=-.001 and top1 delta>=-.005",
        "note": "Candidate universe unchanged. Training positives used only in training, never inserted into evaluation pools. One-query validation is a controlled stress test, not an estimate of competition query-count distribution.",
    }
    path = ROOT / "protocol.json"
    ROOT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == spec
    write_json(path, spec)
    return spec


def single_query(frame):
    parts = []
    for key, group in frame.groupby("inchikey14", sort=True):
        index = int.from_bytes(
            hashlib.sha256(("single:" + key).encode()).digest()[:8], "big"
        ) % len(group)
        parts.append(group.iloc[[index]])
    return pd.concat(parts, ignore_index=True)


def audit():
    raw = pd.read_csv("artifacts/ablation_20260928/coverage_per_molecule.csv")
    prod = pd.read_csv("artifacts/ablation_20260928/production_coverage.csv")
    frame = raw.merge(prod[["key", "in_production_pool", "current"]], on="key")
    frame["failure"] = np.where(
        ~frame.in_production_pool,
        "absent_from_database",
        np.where(~frame.current, "mass_filter_miss", "candidate_present"),
    )
    frame.to_csv(ROOT / "coverage_decomposition.csv", index=False)
    summary = {
        "molecules": len(frame),
        "counts": frame.failure.value_counts().to_dict(),
        "by_source": frame.groupby(["failure", "source"])
        .size()
        .reset_index(name="molecules")
        .to_dict("records"),
        "oracle_mrr_upper_bound_current_pool": float(frame.current.mean()),
        "oracle_mrr_upper_bound_ignoring_mass_filter": float(
            frame.in_production_pool.mean()
        ),
        "scope": "previous development proxy, not competition hidden distribution; no external library expansion or truth-based candidate insertion",
    }
    write_json(ROOT / "coverage_audit.json", summary)
    return summary


def prepare_controls(split):
    if split == "matcheddev" and not (ROOT / f"{split}.parquet").exists():
        single_query(pd.read_parquet(OLD / "dev.parquet")).to_parquet(
            ROOT / f"{split}.parquet", index=False
        )
    previous = ablation.ROOT
    ablation.ROOT = ROOT
    try:
        for mode in ["unknown", "known"]:
            ablation.build_records(split, mode)
    finally:
        ablation.ROOT = previous


@torch.inference_mode()
def latent_cache(split, device):
    out = ROOT / f"encoder_{split}.npz"
    if out.exists():
        return np.load(out)
    model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
    model.to(device).eval()
    if split == "train60k":
        directory = SCALE / "features/train60k"
    else:
        directory = ROOT / "features" / split
        if not (directory / "complete.json").exists():
            cache_features(
                pd.read_parquet(ROOT / f"{split}.parquet"),
                checkpoint["preprocessing"],
                directory,
            )
    dataset = SelectedFeatures(directory, model.input_names, target=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=2
    )
    latents = []
    probabilities = []
    for batch in loader:
        x = torch.cat([batch["hist"], batch["meta"], batch["loss"]], -1).to(device)
        z = model.encoder(x)
        latents.append(z.cpu().numpy())
        probabilities.append(model.head(z).sigmoid().cpu().numpy())
    np.savez(
        out, latent=np.concatenate(latents), probability=np.concatenate(probabilities)
    )
    del model
    gc.collect()
    return np.load(out)


def train_data():
    path = ROOT / "training.joblib"
    if path.exists():
        return joblib.load(path)
    frame = pd.read_parquet(SCALE / "train60k.parquet")
    catalog = pd.read_parquet(OLD / "catalog.parquet").set_index("inchikey14")
    groups = frame.groupby("inchikey14", sort=True).indices
    keys = list(groups)
    smiles = [frame.iloc[groups[k][0]].normalized_smiles for k in keys]
    masses = catalog.loc[keys, "mass"].to_numpy()
    assert set(catalog.loc[keys, "split"]) == {"train"}
    fps = np.stack([fingerprint(s) for s in smiles]).astype(np.float32)
    graphs = []
    for i, s in enumerate(smiles):
        graphs.append(graph(s))
        if (i + 1) % 10000 == 0:
            print("graphs", i + 1, flush=True)
    order = np.argsort(masses)
    sorted_mass = masses[order]
    negatives = np.empty((len(keys), 128), np.int64)
    inside = []
    for i, m in enumerate(masses):
        pos = np.searchsorted(sorted_mass, m)
        candidates = order[max(0, pos - 140) : min(len(keys), pos + 141)]
        candidates = candidates[candidates != i]
        candidates = candidates[
            np.argsort(np.abs(masses[candidates] - m), kind="stable")
        ][:128]
        assert len(candidates) == 128 and i not in candidates
        negatives[i] = candidates
        inside.append(
            int((np.abs(masses[candidates] - m) <= max(m * 35e-6, 0.006)).sum())
        )
    result = {
        "keys": keys,
        "rows": [np.asarray(groups[k]) for k in keys],
        "fps": fps,
        "graphs": graphs,
        "negatives": negatives,
        "inside": np.asarray(inside),
        "masses": masses,
    }
    joblib.dump(result, path, compress=0)
    write_json(
        ROOT / "training_manifest.json",
        {
            "molecules": len(keys),
            "spectra": len(frame),
            "train_only": True,
            "molecules_with_same_mass_negative": int((np.asarray(inside) > 0).sum()),
            "encoder_sha256": sha(ENCODER),
        },
    )
    return result


class Evaluation:
    def __init__(self, split):
        self.split = split
        self.frame = pd.read_parquet(ROOT / f"{split}.parquet")
        assert self.frame.inchikey14.is_unique
        self.positions = {k: i for i, k in enumerate(self.frame.inchikey14)}
        self.records = {
            m: json.loads((ROOT / f"{split}_{m}_records.json").read_text())
            for m in ["unknown", "known"]
        }
        cat = pd.read_parquet(OLD / "catalog.parquet")
        pools = [build_candidates(cat, COCONUT)]
        observed = set(
            pd.read_parquet(
                ROOT / "reference" / f"{split}_known" / "rows.parquet",
                columns=["inchikey14"],
            ).inchikey14
        )
        knowncat = cat[
            ((cat.split == "train") | cat.inchikey14.isin(self.frame.inchikey14))
            & cat.inchikey14.isin(observed)
        ]
        pools.append(build_candidates(knowncat, COCONUT, final=True))
        # Preserve each scenario's actual library/COCONUT structure representation.
        # In particular, never replace unknown candidates with held-out query SMILES.
        self.ids = {}
        structures = {}
        for mode, pool in zip(["unknown", "known"], pools):
            lookup = pool.set_index("inchikey14").normalized_smiles
            needed = {k for r in self.records[mode] for k in r["available"]["union35"]}
            for key in sorted(needed):
                smi = lookup[key]
                identity = (key, smi)
                if identity not in structures:
                    structures[identity] = len(structures)
                self.ids[mode, key] = structures[identity]
        self.keys = list(structures)
        path = ROOT / f"{split}_structures.joblib"
        if path.exists():
            saved = joblib.load(path)
            assert saved["keys"] == self.keys
            self.fps = saved["fps"]
            self.graphs = saved["graphs"]
        else:
            smiles = [smi for key, smi in self.keys]
            self.fps = np.stack([fingerprint(s) for s in smiles]).astype(np.float32)
            self.graphs = [graph(s) for s in smiles]
            joblib.dump(
                {"keys": self.keys, "fps": self.fps, "graphs": self.graphs},
                path,
                compress=0,
            )
        cached = latent_cache(split, "cpu")
        self.latent = cached["latent"]
        self.prob = cached["probability"]
        self.old = {}
        for mode, records in self.records.items():
            for r in records:
                keys = r["available"]["union35"]
                indices = [self.ids[mode, k] for k in keys]
                self.old[mode, r["key"]] = neural_rank(
                    self.prob[self.positions[r["key"]]], keys, self.fps[indices]
                )

    @torch.inference_mode()
    def ranks(self, model, device):
        model.to(device).eval()
        emb = []
        for start in range(0, len(self.keys), 256):
            fp = torch.from_numpy(self.fps[start : start + 256]).to(device)
            graphs = (
                batch_graphs(self.graphs[start : start + 256], device)
                if model.architecture == "graph"
                else None
            )
            emb.append(model.encode_molecules(fp, graphs).cpu().numpy())
        emb = np.concatenate(emb)
        query = (
            model.encode_spectra(torch.from_numpy(self.latent).to(device)).cpu().numpy()
        )
        output = {}
        for mode, records in self.records.items():
            for r in records:
                key = r["key"]
                candidates = r["available"]["union35"]
                score = (
                    emb[[self.ids[mode, k] for k in candidates]]
                    @ query[self.positions[key]]
                )
                output[mode, key] = [
                    candidates[j]
                    for j in sorted(
                        range(len(candidates)), key=lambda j: (-score[j], candidates[j])
                    )
                ]
        return output

    @torch.inference_mode()
    def deployment_ranks(self, model):
        """Final acceptance uses the same per-group CPU path as deployment."""
        from casmi_ml.direct_models import score_group
        from casmi_ml.inference import group_probability

        encoder, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
        model.cpu().eval()
        output = {}
        probabilities = {}
        for mode, records in self.records.items():
            for i, r in enumerate(records):
                key = r["key"]
                keys = r["available"]["union35"]
                group = self.frame.iloc[[self.positions[key]]]
                ids = [self.ids[mode, k] for k in keys]
                fps = self.fps[ids]
                candidates = pd.DataFrame(
                    {
                        "inchikey14": keys,
                        "normalized_smiles": [self.keys[j][1] for j in ids],
                    }
                )
                if key not in probabilities:
                    probabilities[key] = group_probability(
                        encoder, group, checkpoint["preprocessing"]
                    )
                self.old[mode, key] = neural_rank(probabilities[key], keys, fps)
                scores = score_group(
                    model, encoder, group, checkpoint["preprocessing"], candidates, fps
                )
                output[mode, key] = [
                    keys[j]
                    for j in sorted(
                        range(len(keys)), key=lambda j: (-scores[j], keys[j])
                    )
                ]
                if (i + 1) % 500 == 0:
                    print("deployment-exact ranking", mode, i + 1, flush=True)
        return output

    def score(self, ranks=None, weight=0.0, mode="unknown", raw=False):
        predictions = {}
        pools = {}
        for r in self.records[mode]:
            key = r["key"]
            if mode == "known" and not r["known"]:
                continue
            pools[key] = list(
                set(r["available"]["union35"]) | set(r["available"]["coconut15"])
            )
            historical = r["rankings"]["coconut15"]
            old = self.old[mode, key]
            neural = (
                old
                if not weight
                else rrf([old, ranks[mode, key]], [1 - weight, weight])
            )
            if raw:
                predictions[key] = neural
            elif r["confidence"] >= 0.5:
                predictions[key] = historical
            else:
                predictions[key] = low_confidence_rank(
                    historical, r["current_full"], neural, 0.75
                )
        return metrics(predictions, pools)


def train(architecture, spec, data, evaluation):
    out = ROOT / "runs" / architecture
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    configure(42, 4)
    device = "cuda"
    model = DirectRanker(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    latent = latent_cache("train60k", device)["latent"]
    rng = np.random.default_rng(42)
    best = -1
    stale = 0
    history = []
    start = time.monotonic()
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        losses = []
        for ids in np.array_split(
            rng.permutation(len(data["keys"])),
            int(np.ceil(len(data["keys"]) / spec["batch_size"])),
        ):
            positive = np.array([rng.choice(data["rows"][i]) for i in ids])
            negative = []
            for i in ids:
                # Prefer actual isobaric candidates; supplement with closest masses when sparse.
                width = max(16, int(data["inside"][i]))
                negative.append(
                    rng.choice(data["negatives"][i, :width], 15, replace=False)
                )
            chosen = np.column_stack([ids, np.stack(negative)])
            unique, inverse = np.unique(chosen, return_inverse=True)
            fp = torch.from_numpy(data["fps"][unique]).to(device)
            graphs = (
                batch_graphs([data["graphs"][i] for i in unique], device)
                if architecture == "graph"
                else None
            )
            z = torch.from_numpy(latent[positive]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                z,
                fp,
                graphs,
                torch.from_numpy(inverse.reshape(chosen.shape)).to(device),
            )
            loss = F.cross_entropy(
                logits, torch.zeros(len(ids), dtype=torch.long, device=device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        ranks = evaluation.ranks(model, device)
        reports = []
        baseline = {m: evaluation.score(mode=m)[0] for m in ["unknown", "known"]}
        for weight in spec["neural_blend_weights"]:
            result = {
                m: evaluation.score(ranks, weight, m)[0] for m in ["unknown", "known"]
            }
            eligible = (
                result["known"]["mrr25"] >= baseline["known"]["mrr25"] - 0.001
                and result["known"]["top1"] >= baseline["known"]["top1"] - 0.005
            )
            reports.append({"weight": weight, **result, "eligible": eligible})
        eligible = [r for r in reports if r["eligible"]]
        winner = (
            max(eligible, key=lambda r: r["unknown"]["mrr25"]) if eligible else None
        )
        value = winner["unknown"]["mrr25"] if winner else -1
        raw = evaluation.score(ranks, 1.0, raw=True)[0]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "raw_unknown": raw,
            "candidates": reports,
            "seconds": time.monotonic() - start,
        }
        history.append(row)
        write_json(out / "history.json", history)
        print(
            architecture,
            epoch,
            "loss",
            row["loss"],
            "raw",
            raw["mrr25"],
            "routed",
            value,
            "seconds",
            round(row["seconds"]),
            flush=True,
        )
        if value > best:
            best = value
            stale = 0
            torch.save(
                {
                    "architecture": architecture,
                    "state_dict": model.cpu().state_dict(),
                    "encoder_sha256": sha(ENCODER),
                    "epoch": epoch,
                    "winner": winner,
                },
                out / "model.pt",
            )
            model.to(device)
        else:
            stale += 1
        if epoch >= spec["minimum_epochs"] and stale >= spec["patience"]:
            break
    result = {
        "architecture": architecture,
        "best_unknown_mrr": best,
        "checkpoint": str(out / "model.pt"),
        "seconds": time.monotonic() - start,
        "baseline": baseline,
    }
    write_json(out / "result.json", result)
    return result


def select(evaluation, spec):
    path = ROOT / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    baseline = {m: evaluation.score(mode=m)[0] for m in ["unknown", "known"]}
    results = []
    for name in spec["representations"]:
        pathmodel = ROOT / "runs" / name / "model.pt"
        if not pathmodel.exists():
            continue
        checkpoint = torch.load(pathmodel, map_location="cpu", weights_only=True)
        model = DirectRanker(name)
        model.load_state_dict(checkpoint["state_dict"])
        ranks = evaluation.ranks(model, "cpu")
        for weight in spec["neural_blend_weights"]:
            reports = {
                m: evaluation.score(ranks, weight, m)[0] for m in ["unknown", "known"]
            }
            eligible = (
                reports["unknown"]["mrr25"] > baseline["unknown"]["mrr25"]
                and reports["known"]["mrr25"] >= baseline["known"]["mrr25"] - 0.001
                and reports["known"]["top1"] >= baseline["known"]["top1"] - 0.005
            )
            results.append(
                {
                    "architecture": name,
                    "weight": weight,
                    "checkpoint": str(pathmodel),
                    "checkpoint_sha256": sha(pathmodel),
                    **reports,
                    "eligible": eligible,
                }
            )
    eligible = [r for r in results if r["eligible"]]
    selected = max(eligible, key=lambda r: r["unknown"]["mrr25"]) if eligible else None
    report = {
        "winner": selected,
        "baseline": baseline,
        "comparisons": results,
        "precision": "CPU float32",
        "protocol_sha256": sha(ROOT / "protocol.json"),
    }
    write_json(path, report)
    print("SELECTION", json.dumps(report), flush=True)
    return report


def fresh_data():
    if (ROOT / "fresh.parquet").exists():
        return
    used = set()
    paths = []
    for directory, names in [
        (
            OLD,
            ["train", "dev", "holdout", "fresh_holdout", "diagnostic", "final_train"],
        ),
        (Path("artifacts/ablation_20260928"), ["fresh"]),
        (SCALE, ["fresh", "train60k"]),
        (Path("artifacts/router_20260929"), ["fresh"]),
    ]:
        for name in names:
            path = directory / f"{name}.parquet"
            paths.append(str(path))
            used.update(pd.read_parquet(path, columns=["inchikey14"]).inchikey14)
    catalog = pd.read_parquet(OLD / "catalog.parquet")
    keys = set(
        catalog.loc[
            (catalog.split == "holdout") & ~catalog.inchikey14.isin(used), "inchikey14"
        ].head(2000)
    )
    assert len(keys) == 2000 and not keys & used
    frame = single_query(sample_keys(keys, 20261003))
    frame.to_parquet(ROOT / "fresh.parquet", index=False)
    write_json(
        ROOT / "fresh_manifest.json",
        {
            "molecules": len(keys),
            "spectra": len(frame),
            "prior_overlap": 0,
            "excluded_paths": paths,
            "selection_sha256": sha(ROOT / "selection.json"),
            "sha256": sha(ROOT / "fresh.parquet"),
        },
    )


def evaluate(selection):
    path = ROOT / "fresh_report.json"
    if path.exists():
        return json.loads(path.read_text())
    winner = selection["winner"]
    if not winner:
        report = {
            "accepted": False,
            "reason": "No development improvement; no new holdout exposed",
        }
        write_json(path, report)
        return report
    fresh_data()
    prepare_controls("fresh")
    evaluation = Evaluation("fresh")
    ckpt = torch.load(winner["checkpoint"], map_location="cpu", weights_only=True)
    model = DirectRanker(winner["architecture"])
    model.load_state_dict(ckpt["state_dict"])
    ranks = evaluation.deployment_ranks(model)
    reports = {}
    for mode in ["unknown", "known"]:
        result, rows = evaluation.score(ranks, winner["weight"], mode)
        baseline, old = evaluation.score(mode=mode)
        result.update(baseline=baseline, difference=bootstrap_difference(rows, old))
        rows.to_csv(ROOT / f"fresh_{mode}_selected.csv", index=False)
        old.to_csv(ROOT / f"fresh_{mode}_baseline.csv", index=False)
        reports[mode] = result
    reports["accepted"] = (
        reports["unknown"]["difference"]["ci95"][0] > 0
        and reports["known"]["difference"]["difference"] >= -0.001
        and reports["known"]["top1"] >= reports["known"]["baseline"]["top1"] - 0.005
    )
    write_json(path, reports)
    print("FRESH", json.dumps(reports), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    args = parser.parse_args()
    configure(42, 4)
    spec = protocol()
    if args.stage == "prepare":
        audit()
        prepare_controls("matcheddev")
        return
    if args.stage == "train":
        data = train_data()
        latent_cache("train60k", "cuda")
        evaluation = Evaluation("matcheddev")
        for name in spec["representations"]:
            train(name, spec, data, evaluation)
        select(evaluation, spec)
    else:
        evaluate(json.loads((ROOT / "selection.json").read_text()))


if __name__ == "__main__":
    main()
