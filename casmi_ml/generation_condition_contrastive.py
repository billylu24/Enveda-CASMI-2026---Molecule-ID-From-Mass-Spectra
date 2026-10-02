"""Training-only same-mass condition contrast for a frozen-architecture decoder."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.nn import functional as F

from casmi_ml.data import write_json
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


def nearest_mass_negatives(masses, keys):
    masses = np.asarray(masses, dtype=float)
    if (
        len(masses) < 2
        or len(masses) != len(keys)
        or len(set(keys)) != len(keys)
        or not np.isfinite(masses).all()
    ):
        raise ValueError("Require distinct molecules with finite masses")
    order = sorted(range(len(keys)), key=lambda i: (masses[i], keys[i]))
    result = np.empty(len(keys), dtype=np.int64)
    for position, index in enumerate(order):
        adjacent = (
            order[max(0, position - 1) : position] + order[position + 1 : position + 2]
        )
        result[index] = min(
            adjacent, key=lambda j: (abs(masses[j] - masses[index]), keys[j])
        )
    return result


def sequence_nll(logits, targets):
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=0
    )
    return token_loss.sum(1) / (targets != 0).sum(1).clamp_min(1)


def train(output, epochs=3, seconds=3600, contrast_weight=0.1):
    if not np.isfinite(contrast_weight) or not 0 <= contrast_weight <= 0.1:
        raise ValueError("Contrast weight must be in[0,.1]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        "artifacts/research_loop/rounds/0020_decoder_single_control/model.pt"
    )
    spec = {
        "version": 1,
        "source_sha256": digest(Path(__file__)),
        "initial_checkpoint_sha256": digest(checkpoint),
        "train_sha256": digest(TRAIN),
        "epochs": epochs,
        "seconds": seconds,
        "seed": 42,
        "lr": 1e-5,
        "contrast_weight": contrast_weight,
        "margin_nats_per_token": 0.2,
        "negative": "nearest exact-structure-mass distinct training molecule; actual randomly selected spectrum condition; deterministic mass/key ties",
        "loss": "positive token CE +contrast_weight mean relu(0.2+positive per-sequence NLL-negative per-sequence NLL)",
        "formula_frozen": True,
        "encoder_frozen": True,
        "training_scope": "original60000 training molecules only; one positive target per molecule per epoch; two teacher-forced forwards",
        "selection": "fixed final epoch; no development likelihood selection",
        "holdout_used": False,
        "adversarial_training": False,
    }
    freeze(output / "protocol.json", spec)
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    configure(42, threads=4)
    budget = StageBudget(output, "generation", "condition_contrast", seconds)
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
        groups = frame.groupby("inchikey14", sort=True).indices
        keys = list(groups)
        dev_keys = set(
            pd.read_parquet(
                ROOT / "researchdev.parquet", columns=["inchikey14"]
            ).inchikey14
        )
        if len(keys) != 60000 or set(keys) & dev_keys:
            raise ValueError("Training/development isolation failed")
        group_ids = list(groups.values())
        masses = np.array(
            [
                Descriptors.ExactMolWt(
                    Chem.MolFromSmiles(frame.normalized_smiles.iloc[v[0]])
                )
                for v in group_ids
            ]
        )
        negative = nearest_mass_negatives(masses, keys)
        gaps = abs(masses - masses[negative])
        values = np.load(conditions(ROOT, "train60k") / "condition.npy", mmap_mode="r")
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-5, weight_decay=1e-4)
        rng = np.random.default_rng(42)
        history = []
        for epoch in range(1, epochs + 1):
            if not budget.checkpoint():
                break
            selected = np.array([int(rng.choice(ids)) for ids in group_ids])
            dataset = GenerationDataset(frame, values, vocabulary, selected.tolist())
            row_to_group = {int(row): i for i, row in enumerate(selected)}
            usable = np.array(dataset.indexes)
            rng.shuffle(usable)
            decoder.train()
            sums = np.zeros(3)
            batches = 0
            for start in range(0, len(usable), 64):
                if not budget.checkpoint():
                    break
                rows = usable[start : start + 64]
                batch = [
                    (
                        torch.from_numpy(np.array(values[row])),
                        torch.tensor(dataset.sequences[row]),
                        torch.zeros(1, dtype=torch.long),
                    )
                    for row in rows
                ]
                positive, tokens, _ = collate(batch)
                indices = [selected[negative[row_to_group[int(row)]]] for row in rows]
                wrong = torch.from_numpy(np.array(values[indices])).to(device)
                positive, tokens = positive.to(device), tokens.to(device)
                with torch.no_grad():
                    positive = torch.cat([positive, formula.soft(positive)], 1)
                    wrong = torch.cat([wrong, formula.soft(wrong)], 1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = decoder(tokens[:, :-1], positive)
                    wrong_logits = decoder(tokens[:, :-1], wrong)
                    ce = F.cross_entropy(
                        logits.transpose(1, 2), tokens[:, 1:], ignore_index=0
                    )
                    positive_nll = sequence_nll(logits, tokens[:, 1:])
                    negative_nll = sequence_nll(wrong_logits, tokens[:, 1:])
                    contrast = F.relu(0.2 + positive_nll - negative_nll).mean()
                    loss = ce + contrast_weight * contrast
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite condition contrast loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                sums += [
                    float(ce.detach()),
                    float(contrast.detach()),
                    float((negative_nll - positive_nll).mean().detach()),
                ]
                batches += 1
            complete = batches == (len(usable) + 63) // 64
            entry = {
                "epoch": epoch,
                "positive_token_ce": float(sums[0] / max(batches, 1)),
                "contrast_loss": float(sums[1] / max(batches, 1)),
                "training_nll_gap": float(sums[2] / max(batches, 1)),
                "batches": batches,
                "training_molecules": len(usable),
                "complete": complete,
                "seconds": time.monotonic() - started,
            }
            history.append(entry)
            write_json(output / "history.json", history)
            print(entry, flush=True)
            if not complete:
                break
        complete = len(history) == epochs and history[-1]["complete"]
        if complete:
            saved["decoder"] = {
                k: v.cpu().clone() for k, v in decoder.state_dict().items()
            }
            saved["condition_contrast_training"] = spec
            saved["initial_report"] = saved.pop("report", None)
            saved["report"] = {
                "scope": "fixed final epoch contrast control",
                "history": history,
            }
            torch.save(saved, output / "model.pt")
        report = {
            "diagnostic_only": True,
            "complete": complete,
            "history": history,
            "seconds": time.monotonic() - started,
            "training_molecules": len(keys),
            "training_development_key_overlap": 0,
            "negatives_within_mass_window": int(
                (gaps <= np.maximum(0.006, masses * 35e-6)).sum()
            ),
            "median_negative_mass_gap": float(np.median(gaps)),
            "checkpoint_sha256": digest(output / "model.pt") if complete else None,
            "independent_acceptance": False,
            "adversarial_training": False,
            "performance_acceptance": "fixed200 structure pilot against current decoder; full2000 ranking required before publication",
        }
        write_json(output / "report.json", report)
        return report
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--contrast-weight", type=float, default=0.1)
    p.add_argument("--seconds", type=float, default=3600)
    a = p.parse_args()
    if a.epochs < 1 or not 0 < a.seconds <= 86400:
        p.error("Positive epochs and budget at most24h required")
    print(json.dumps(train(a.output, a.epochs, a.seconds, a.contrast_weight), indent=2))


if __name__ == "__main__":
    main()
