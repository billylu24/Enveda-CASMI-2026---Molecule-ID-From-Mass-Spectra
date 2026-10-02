"""Matched decoder fine-tuning controls for training/deployment condition averaging."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from casmi_ml.data import write_json
from casmi_ml.generation_augmentation import enumerated_target
from casmi_ml.generation_experiment import (
    GenerationDataset,
    collate,
    conditions,
    load_model,
)
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.training import configure


def condition_values(values, group_indices, averaging):
    if averaging == "single":
        return values
    if averaging != "mean":
        raise ValueError("Unknown condition averaging")
    result = np.empty_like(values)
    for indices in group_indices:
        result[indices] = np.asarray(values[indices]).mean(0)
    return result


def train(
    output,
    averaging,
    epochs=3,
    seconds=3600,
    targets="canonical",
    learning_rate=3e-5,
    initial_checkpoint=None,
):
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 0.001:
        raise ValueError("Finite learning rate in(0,.001] required")
    if targets not in ["canonical", "randomized"]:
        raise ValueError("Unknown target augmentation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure(42, threads=4)
    checkpoint = Path(initial_checkpoint or ROOT / "generation/smiles_42/model.pt")
    spec = {
        "version": 1,
        "source_sha256": digest(Path(__file__)),
        "averaging": averaging,
        "epochs": epochs,
        "seconds": seconds,
        "initial_checkpoint_sha256": digest(checkpoint),
        "train_sha256": digest(TRAIN),
        "lr": learning_rate,
        "seed": 42,
        "formula_frozen": True,
        "encoder_frozen": True,
        "selection": "Fixed final epoch; no development likelihood checkpoint selection",
        "training_scope": "Same frozen 60000 train molecules; one target/gradient per molecule per epoch",
        "holdout_used": False,
    }
    if targets == "randomized":
        spec["target_augmentation"] = {
            "method": "one deterministic randomized SMILES per molecule per epoch",
            "source_sha256": digest(
                Path(__file__).with_name("generation_augmentation.py")
            ),
            "fallback": "original sequence if frozen vocabulary or length rejects augmentation",
        }
    freeze(output / "protocol.json", spec)
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    budget = StageBudget(output, "generation", "decoder_finetune", seconds)
    started = time.monotonic()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.6)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        decoder, formula, vocabulary = load_model(checkpoint, device)
        formula.requires_grad_(False).eval()
        frame = pd.read_parquet(
            TRAIN, columns=["inchikey14", "normalized_smiles", "molecular_formula"]
        )
        groups = list(frame.groupby("inchikey14", sort=True).indices.values())
        values = np.load(conditions(ROOT, "train60k") / "condition.npy", mmap_mode="r")
        values = condition_values(values, groups, averaging)
        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        rng = np.random.default_rng(42)
        history = []
        for epoch in range(1, epochs + 1):
            if not budget.checkpoint():
                break
            indices = [int(rng.choice(ids)) for ids in groups]
            dataset = GenerationDataset(frame, values, vocabulary, indices)
            augmentation = {"changed": 0, "vocabulary_or_length_fallback": 0}
            if targets == "randomized":
                for index in dataset.indexes:
                    row = frame.iloc[index]
                    sequence, stats = enumerated_target(
                        row.normalized_smiles, row.inchikey14, epoch, vocabulary
                    )
                    dataset.sequences[index] = sequence
                    for name in augmentation:
                        augmentation[name] += int(stats[name])
            decoder.train()
            total_loss, batches = 0.0, 0
            for condition, tokens, _ in DataLoader(
                dataset, batch_size=64, shuffle=True, collate_fn=collate
            ):
                if not budget.checkpoint():
                    break
                condition, tokens = condition.to(device), tokens.to(device)
                with torch.no_grad():
                    counts = formula.soft(condition)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = decoder(tokens[:, :-1], torch.cat([condition, counts], 1))
                    loss = F.cross_entropy(
                        logits.transpose(1, 2), tokens[:, 1:], ignore_index=0
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite decoder fine-tuning loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1
            complete = batches == (len(dataset) + 63) // 64
            entry = {
                "epoch": epoch,
                "token_loss": total_loss / max(batches, 1),
                "batches": batches,
                "training_molecules": len(dataset),
                "complete": complete,
                "seconds": time.monotonic() - started,
            }
            if targets == "randomized":
                entry["augmentation"] = augmentation
            history.append(entry)
            write_json(output / "history.json", history)
            print(entry, flush=True)
            if not complete:
                break
        if len(history) == epochs and history[-1]["complete"]:
            saved["decoder"] = {
                k: v.cpu().clone() for k, v in decoder.state_dict().items()
            }
            saved["decoder_finetuning"] = spec
            saved["initial_report"] = saved.pop("report", None)
            saved["report"] = {
                "scope": "Fixed final epoch decoder fine-tuning; original likelihood report is preserved separately",
                "history": history,
            }
            torch.save(saved, output / "model.pt")
        result = {
            "history": history,
            "seconds": time.monotonic() - started,
            "complete": len(history) == epochs and history[-1]["complete"],
            "checkpoint_sha256": digest(output / "model.pt")
            if (output / "model.pt").exists()
            else None,
            "diagnostic_only": True,
            "independent_acceptance": False,
            "performance_acceptance": "Requires fixed-budget development sampling and full paired 2000-molecule structural rankings",
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--averaging", choices=["single", "mean"], required=True)
    p.add_argument(
        "--targets", choices=["canonical", "randomized"], default="canonical"
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seconds", type=float, default=3600)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--initial-checkpoint", type=Path)
    a = p.parse_args()
    if a.epochs < 1 or not 0 < a.seconds <= 86400:
        p.error("Positive epochs and at most24h required")
    print(
        json.dumps(
            train(
                a.output,
                a.averaging,
                a.epochs,
                a.seconds,
                a.targets,
                a.learning_rate,
                a.initial_checkpoint,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
