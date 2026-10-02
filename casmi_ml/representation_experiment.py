"""Bounded 60K peak encoder comparisons; holdout is never used to train/select."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from casmi_ml import ablation
from casmi_ml.chemistry_experiment import ROUTING, candidate_lookup, reference_records
from casmi_ml.data import metadata, write_json
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics, neural_rank
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_models import (
    Distillation,
    PeakEncoder,
    latent_diagnostics,
    masked_loss,
    peak_tokens,
)
from casmi_ml.research_protocol import ROOT, TRAIN, freeze, prepare
from casmi_ml.training import configure


class PeakDataset(Dataset):
    def __init__(self, directory, indexes=None):
        directory = Path(directory)
        self.arrays = {
            k: np.load(directory / f"{k}.npy", mmap_mode="r")
            for k in ["peaks", "mask", "protected", "meta", "target"]
        }
        self.indexes = (
            np.arange(len(self.arrays["mask"]))
            if indexes is None
            else np.asarray(indexes)
        )

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return {
            k: torch.from_numpy(
                np.unpackbits(np.array(v[self.indexes[index]], copy=True))
                if k == "target"
                else np.array(v[self.indexes[index]], copy=True)
            )
            for k, v in self.arrays.items()
        }


def cache(root, split):
    root = Path(root)
    path = TRAIN if split == "train60k" else root / f"{split}.parquet"
    directory = root / "peak_features" / split
    directory.mkdir(parents=True, exist_ok=True)
    prep = json.loads(Path("artifacts/experiment/preprocessing.json").read_text())
    spec = {
        "input_sha256": digest(path),
        "preprocessing": prep,
        "max_peaks": 128,
        "version": 1,
    }
    marker = directory / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text()) != spec:
            raise ValueError("Changed peak feature cache")
        return directory
    frame = pd.read_parquet(path)
    dim = 6 + sum(len(v) + 1 for v in prep["categories"].values())
    shapes = {
        "peaks": (len(frame), 128, 19),
        "mask": (len(frame), 128),
        "protected": (len(frame), 128),
        "meta": (len(frame), dim),
        "target": (len(frame), 256),
    }
    arrays = {
        k: np.lib.format.open_memmap(
            directory / f"{k}.npy",
            mode="w+",
            shape=s,
            dtype=bool
            if k in ["mask", "protected"]
            else np.uint8
            if k == "target"
            else np.float32,
        )
        for k, s in shapes.items()
    }
    for i, row in enumerate(frame.to_dict("records")):
        for k, v in zip(["peaks", "mask", "protected"], peak_tokens(row)):
            arrays[k][i] = v
        arrays["meta"][i] = metadata(row, prep)
        arrays["target"][i] = np.frombuffer(row["fingerprint"], dtype=np.uint8)
        if (i + 1) % 20000 == 0:
            print("peak cache", split, i + 1, flush=True)
    for arr in arrays.values():
        arr.flush()
    frame[["inchikey14", "row_id"]].to_parquet(directory / "rows.parquet", index=False)
    write_json(marker, spec)
    return directory


def molecule_indexes(directory, rng):
    info = pd.read_parquet(Path(directory) / "rows.parquet")
    return [
        int(rng.choice(ids))
        for ids in info.groupby("inchikey14", sort=True).indices.values()
    ]


@torch.inference_mode()
def predict(model, directory, device, latents=False):
    model.eval()
    probabilities, embeddings = [], []
    for batch in DataLoader(PeakDataset(directory), batch_size=128):
        batch = {k: v.to(device) for k, v in batch.items()}
        z = model.encode(batch)
        probabilities.append(model.fingerprint(z).sigmoid().float().cpu().numpy())
        if latents:
            embeddings.append(z.float().cpu().numpy())
    return np.concatenate(probabilities), (
        np.concatenate(embeddings) if latents else None
    )


class Development:
    def __init__(self, root, split="researchdev"):
        self.root, self.split = Path(root), split
        frame = pd.read_parquet(self.root / f"{split}.parquet")
        self.groups = frame.groupby("inchikey14", sort=True).indices
        self.records = {
            m: reference_records(self.root, split, m) for m in ["unknown", "known"]
        }
        self.lookup = {m: candidate_lookup(self.root, split, m) for m in self.records}
        self.fp_cache = {}
        fingerprint_cache = self.root / f"{split}_candidate_fingerprints.joblib"
        from casmi_ml.data import fingerprint

        structures = {
            (mode, k): self.lookup[mode][k]
            for mode, records in self.records.items()
            for r in records
            for k in r["available"]["union35"]
        }
        if fingerprint_cache.exists():
            saved = joblib.load(fingerprint_cache)
            if saved["structures"] != structures:
                raise ValueError("Candidate fingerprint cache representations changed")
            self.fp_cache = saved["fingerprints"]
        else:
            by_smiles = {}
            for cache_key, smiles in structures.items():
                if smiles not in by_smiles:
                    by_smiles[smiles] = fingerprint(smiles)
                self.fp_cache[cache_key] = by_smiles[smiles]
            temp = fingerprint_cache.with_suffix(".tmp")
            joblib.dump(
                {"structures": structures, "fingerprints": self.fp_cache},
                temp,
                compress=3,
            )
            temp.replace(fingerprint_cache)

    def evaluate(self, probability):
        from casmi_ml.data import fingerprint

        reports, raw = {}, {}
        for mode, records in self.records.items():
            rankings, raw_rankings, pools = {}, {}, {}
            for r in records:
                if mode == "known" and not r["known"]:
                    continue
                keys = r["available"]["union35"]
                for key in keys:
                    cache_key = (mode, key)
                    if cache_key not in self.fp_cache:
                        self.fp_cache[cache_key] = fingerprint(self.lookup[mode][key])
                fps = np.array(
                    [self.fp_cache[(mode, k)] for k in keys], dtype=np.float32
                ).reshape(-1, 2048)
                rank = neural_rank(
                    probability[self.groups[r["key"]]].mean(0), keys, fps
                )
                rankings[r["key"]] = ablation.route({**r, "neural": rank}, ROUTING)
                raw_rankings[r["key"]] = rank
                pools[r["key"]] = list(set(keys) | set(r["available"]["coconut15"]))
            reports[mode] = metrics(rankings, pools)[0]
            raw[mode] = metrics(raw_rankings, pools)[0]
        return {"routed": reports, "raw": raw}

    def baseline(self):
        result = {}
        for mode, records in self.records.items():
            chosen = [r for r in records if mode != "known" or r["known"]]
            result[mode] = metrics(
                {r["key"]: ablation.route(r, ROUTING) for r in chosen},
                {
                    r["key"]: list(
                        set(r["available"]["union35"])
                        | set(r["available"]["coconut15"])
                    )
                    for r in chosen
                },
            )[0]
        return result


def train(
    root=ROOT, method="dino", seed=42, pretrain_epochs=20, epochs=30, seconds=86400
):
    root = Path(root)
    if seconds <= 0 or seconds > 86400:
        raise ValueError("Per-stage budget must be in (0,24h]")
    prepare(root)
    configure(seed, threads=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    run = root / "representation" / f"{method}_{seed}_v2"
    run.mkdir(parents=True, exist_ok=True)
    spec = {
        "method": method,
        "seed": seed,
        "width": 256,
        "layers": 4,
        "max_peaks": 128,
        "pretrain_epochs": pretrain_epochs if method != "supervised" else 0,
        "finetune_epochs": epochs,
        "version": 2,
        "early_stopping": "pure_neural_dev_mrr_patience5; deployment_protection_separate",
        "seconds": seconds,
        "training_sha256": digest(TRAIN),
        "dev_sha256": digest(root / "researchdev.parquet"),
        "pretrain_lr": 1e-4,
        "finetune_lr": 3e-4,
        "finetune_rng_seed": seed + 1000,
        "augmentation": "drop_only_weak_peaks;retain_protected;mass_fixed",
    }
    freeze(run / "config.json", spec)
    if (run / "result.json").exists():
        return json.loads((run / "result.json").read_text())
    training, devdir = cache(root, "train60k"), cache(root, "researchdev")
    dim = np.load(training / "meta.npy", mmap_mode="r").shape[1]
    model = PeakEncoder(dim).to(device)
    ssl = Distillation(model).to(device).train() if method == "dino" else None
    optimizer = torch.optim.AdamW(
        ssl.parameters() if ssl else model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    budget = StageBudget(root, "representation", run, seconds)
    start = time.monotonic()
    deadline = start + budget.allowance
    history, rng = [], np.random.default_rng(seed)
    ssl_path = run / "pretrained.pt"
    previous_pretrained = run.parent / f"{method}_{seed}" / "pretrained.pt"
    if method == "masked" and previous_pretrained.exists() and not ssl_path.exists():
        previous = torch.load(
            previous_pretrained, map_location="cpu", weights_only=True
        )
        if any(
            previous["config"][k] != spec[k]
            for k in [
                "method",
                "seed",
                "width",
                "layers",
                "max_peaks",
                "training_sha256",
                "pretrain_epochs",
            ]
        ):
            raise ValueError("Previous pretraining is incompatible")
        ssl_path = previous_pretrained
        freeze(
            run / "pretraining_reuse.json",
            {
                "path": str(ssl_path),
                "sha256": digest(ssl_path),
                "note": "Only finetuning stopping rule changed; pretrained weights reused.",
            },
        )
    if method != "supervised":
        if ssl_path.exists():
            saved = torch.load(ssl_path, map_location=device, weights_only=True)
            model.load_state_dict(saved["state_dict"])
        else:
            for epoch in range(1, pretrain_epochs + 1):
                dataset = PeakDataset(training, molecule_indexes(training, rng))
                total, count = 0.0, 0
                model.train()
                if ssl:
                    ssl.train()
                for batch in DataLoader(dataset, batch_size=128, shuffle=True):
                    if time.monotonic() >= deadline:
                        break
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        loss = ssl.loss(batch) if ssl else masked_loss(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite pretraining loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        ssl.parameters() if ssl else model.parameters(), 1.0
                    )
                    optimizer.step()
                    if ssl:
                        ssl.update_teacher(
                            0.996 + 0.004 * (epoch - 1) / max(pretrain_epochs, 1)
                        )
                    total += float(loss.detach())
                    count += 1
                _, z = predict(model, devdir, device, latents=True)
                diagnostic = latent_diagnostics(z)
                entry = {
                    "phase": "pretrain",
                    "epoch": epoch,
                    "loss": total / max(count, 1),
                    "diagnostic": diagnostic,
                    "seconds": time.monotonic() - start,
                }
                history.append(entry)
                budget.checkpoint()
                write_json(run / "history.json", history)
                print(entry, flush=True)
                if diagnostic["collapsed"] or time.monotonic() >= deadline:
                    break
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": spec,
                    "history": history,
                },
                ssl_path,
            )
            if history[-1]["diagnostic"]["collapsed"]:
                result = {
                    "status": "collapsed",
                    "accepted_for_holdout": False,
                    "history": history,
                }
                write_json(run / "result.json", result)
                return result
    del ssl, optimizer
    configure(seed + 1000, threads=4)
    rng = np.random.default_rng(seed + 1000)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    development = Development(root)
    baseline = development.baseline()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, stale, best_epoch, convergence_best = -1.0, 0, 0, -1.0
    for epoch in range(1, epochs + 1):
        if time.monotonic() >= deadline:
            break
        model.train()
        dataset = PeakDataset(training, molecule_indexes(training, rng))
        total, seen = 0.0, 0
        for batch in DataLoader(dataset, batch_size=128, shuffle=True):
            if time.monotonic() >= deadline:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = F.binary_cross_entropy_with_logits(
                    model(batch), batch["target"].float()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite fingerprint loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            seen += 1
        p, z = predict(model, devdir, device, latents=True)
        report = development.evaluate(p)
        unknown, known = report["routed"]["unknown"], report["routed"]["known"]
        eligible = (
            known["mrr25"] >= baseline["known"]["mrr25"] - 0.001
            and known["top1"] >= baseline["known"]["top1"] - 0.005
        )
        # Save at least one deployable checkpoint, but acceptance requires improvement.
        score = unknown["mrr25"] if eligible else -1.0
        entry = {
            "phase": "finetune",
            "epoch": epoch,
            "loss": total / max(seen, 1),
            "report": report,
            "eligible": eligible,
            "diagnostic": latent_diagnostics(z),
            "seconds": time.monotonic() - start,
        }
        history.append(entry)
        budget.checkpoint()
        write_json(run / "history.json", history)
        print(entry, flush=True)
        if score > best or best_epoch == 0:
            best = score
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": spec,
                    "metadata_dim": dim,
                    "report": report,
                    "baseline": baseline,
                    "preprocessing": json.loads(
                        Path("artifacts/experiment/preprocessing.json").read_text()
                    ),
                },
                run / "model.pt",
            )
        convergence_score = report["raw"]["unknown"]["mrr25"]
        if convergence_score > convergence_best:
            convergence_best = convergence_score
            stale = 0
        else:
            stale += 1
        if epoch >= 8 and stale >= 5:
            break
    result = {
        "status": "complete" if best_epoch else "budget_before_finetune",
        "best_epoch": best_epoch,
        "accepted_for_holdout": best > baseline["unknown"]["mrr25"],
        "best_routed_mrr": best,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else None,
        "seconds": time.monotonic() - start,
        "history": history,
        "baseline": baseline,
    }
    budget.close()
    write_json(run / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=["supervised", "masked", "dino"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seconds", type=float, default=86400)
    a = p.parse_args()
    train(a.root, a.method, a.seed, a.pretrain_epochs, a.epochs, a.seconds)


if __name__ == "__main__":
    main()
