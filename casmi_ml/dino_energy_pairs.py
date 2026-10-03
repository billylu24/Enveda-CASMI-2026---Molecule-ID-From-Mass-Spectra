"""Matched DINO controls: same spectrum versus actual same-ion cross-energy spectra."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.representation_experiment import Development, PeakDataset, cache, predict
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_models import Distillation, PeakEncoder, latent_diagnostics
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.training import configure


def energy_signature(value):
    if value is None:
        return None
    values = np.asarray(value, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        return None
    return tuple(sorted(set(values.tolist())))


def pairing_table(frame):
    energies = [energy_signature(value) for value in frame.collision_energy_ev]
    groups = frame.groupby(
        ["inchikey14", "adduct", "instrument_type"], dropna=False, sort=True
    ).indices
    table = [np.empty(0, dtype=np.int64) for _ in range(len(frame))]
    for ids in groups.values():
        first_row = frame.iloc[int(ids[0])]
        if any(
            pd.isna(first_row[field])
            or str(first_row[field]).strip().lower()
            in {"", "unknown", "none", "nan", "n/a"}
            for field in ("adduct", "instrument_type")
        ):
            continue
        for i in ids:
            if energies[i] is not None:
                table[i] = np.asarray(
                    [
                        j
                        for j in ids
                        if energies[j] is not None and energies[j] != energies[i]
                    ],
                    dtype=np.int64,
                )
    return table


class PairDataset(Dataset):
    def __init__(self, directory, first, second):
        self.first = PeakDataset(directory, first)
        self.second = PeakDataset(directory, second)

    def __len__(self):
        return len(self.first)

    def __getitem__(self, index):
        return self.first[index], self.second[index]


def choose_pairs(groups, table, rng):
    first, second = [], []
    for ids in groups.values():
        i = int(rng.choice(ids))
        choices = table[i]
        first.append(i)
        second.append(int(rng.choice(choices)) if len(choices) else i)
    return np.asarray(first), np.asarray(second)


def run(output, cross_energy=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "source_sha256": digest(Path(__file__)),
        "model_source_sha256": digest("casmi_ml/research_models.py"),
        "training_sha256": digest(TRAIN),
        "development_sha256": digest(ROOT / "researchdev.parquet"),
        "cross_energy": cross_energy,
        "seed": 42,
        "width": 256,
        "layers": 4,
        "pretrain_epochs": 20,
        "finetune_epochs": 30,
        "batch_size": 128,
        "pretrain_lr": 1e-4,
        "finetune_lr": 3e-4,
        "weight_decay": 1e-4,
        "positive_pairs": "Same training molecule key, exact adduct and instrument_type, distinct finite nonnegative eV signature; no eligible partner uses same row",
        "control": "Consume identical first/second row sampling RNG and augmentation forwards; control uses first row for both views",
        "finetune_sampling": "Fresh numpy seed1042; equally molecule-weighted one spectrum per epoch; same fixed steps and initializations",
        "selection": "Fixed final epoch30; development labels used only for final diagnostic evaluation",
        "stage_seconds": 86400,
        "new_generation": False,
        "adversarial_training": False,
        "cohort": "repeated_development",
        "independent_acceptance": False,
        "scope": "Representation diagnostic only on original candidate pool; cannot qualify as improvement over combined0062",
    }
    freeze(output / "protocol.json", spec)
    configure(42, threads=4)
    frame = pd.read_parquet(
        TRAIN,
        columns=[
            "inchikey14",
            "row_id",
            "adduct",
            "instrument_type",
            "collision_energy_ev",
        ],
    )
    groups = frame.groupby("inchikey14", sort=True).indices
    devkeys = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    if len(groups) != 60000 or set(groups) & devkeys:
        raise ValueError("Training/development isolation or60K scope failed")
    table = pairing_table(frame)
    training, dev = cache(ROOT, "train60k"), cache(ROOT, "researchdev")
    if (
        not pd.read_parquet(training / "rows.parquet")
        .reset_index(drop=True)
        .equals(frame[["inchikey14", "row_id"]].reset_index(drop=True))
    ):
        raise ValueError("Cached peak row mapping differs from original training rows")
    dimension = np.load(training / "meta.npy", mmap_mode="r").shape[1]
    evaluator = Development(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("Matched DINO control requires GPU")
    budget = StageBudget(output, "representation", "dino_fixed50epochs", 86400)
    started = time.monotonic()
    try:
        device = torch.device("cuda")
        model = PeakEncoder(dimension).to(device)
        ssl = Distillation(model).to(device)
        optimizer = torch.optim.AdamW(ssl.parameters(), lr=1e-4, weight_decay=1e-4)
        rng = np.random.default_rng(42)
        history = []
        for epoch in range(1, 21):
            first, second = choose_pairs(groups, table, rng)
            paired = int((first != second).sum())
            dataset = PairDataset(training, first, second)
            ssl.train()
            total, count = 0.0, 0
            for batch, partner in DataLoader(dataset, batch_size=128, shuffle=True):
                if not budget.checkpoint():
                    raise TimeoutError("DINO stage budget exhausted")
                batch = {k: v.to(device) for k, v in batch.items()}
                partner = {k: v.to(device) for k, v in partner.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = ssl.loss(batch, partner if cross_energy else batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DINO loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ssl.parameters(), 1.0)
                optimizer.step()
                ssl.update_teacher(0.996 + 0.004 * (epoch - 1) / 20)
                total += float(loss.detach())
                count += 1
            entry = {
                "phase": "pretrain",
                "epoch": epoch,
                "loss": total / count,
                "available_cross_energy_pairs": paired,
                "seconds": time.monotonic() - started,
            }
            history.append(entry)
            write_json(output / "history.json", history)
            print(json.dumps(entry), flush=True)
        # Reset both arms to the same finetuning sampling/dropout/permutation RNG.
        configure(1042, threads=4)
        rng = np.random.default_rng(1042)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for epoch in range(1, 31):
            first = [int(rng.choice(ids)) for ids in groups.values()]
            model.train()
            total, count = 0.0, 0
            for batch in DataLoader(
                PeakDataset(training, first), batch_size=128, shuffle=True
            ):
                if not budget.checkpoint():
                    raise TimeoutError("DINO stage budget exhausted")
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.binary_cross_entropy_with_logits(
                        model(batch), batch["target"].float()
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite fingerprint finetuning loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                count += 1
            entry = {
                "phase": "finetune",
                "epoch": epoch,
                "loss": total / count,
                "seconds": time.monotonic() - started,
            }
            history.append(entry)
            write_json(output / "history.json", history)
            print(json.dumps(entry), flush=True)
        probability, latent = predict(model, dev, device, latents=True)
        report = evaluator.evaluate(probability)
        checkpoint = output / "model.pt"
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "metadata_dim": dimension,
                "config": spec,
            },
            checkpoint,
        )
        result = {
            "diagnostic_only": True,
            "representation": report,
            "baseline": evaluator.baseline(),
            "diagnostic": latent_diagnostics(latent),
            "cross_energy": cross_energy,
            "train_molecules": 60000,
            "train_spectra": len(frame),
            "train_development_overlap": 0,
            "eligible_cross_energy_rows": sum(bool(len(v)) for v in table),
            "seconds": time.monotonic() - started,
            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "checkpoint_sha256": digest(checkpoint),
            "independent_acceptance": False,
            "scope": spec["scope"],
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cross-energy", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.cross_energy), indent=2))


if __name__ == "__main__":
    main()
