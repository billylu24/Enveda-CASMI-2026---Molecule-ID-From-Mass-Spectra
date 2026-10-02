"""Frozen decoder control: fine-tune predicted element counts with mass consistency."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from baseline import ATOMIC_MASS
from casmi_ml.data import write_json
from casmi_ml.generation_experiment import (
    ELEMENTS,
    FormulaPredictor,
    conditions,
    formula_counts,
)
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.training import configure


@torch.inference_mode()
def evaluate(model, x, y, masses, device):
    model.eval()
    correct, top5, counts, total, nll, error = 0, 0, 0, 0, 0.0, 0.0
    for start in range(0, len(y), 256):
        z = torch.from_numpy(np.array(x[start : start + 256])).to(device)
        label = torch.from_numpy(y[start : start + 256]).to(device)
        logits = model(z)
        predicted = logits.argmax(-1)
        correct += int((predicted == label).all(1).sum())
        top5 += int(
            (logits.topk(5, dim=-1).indices == label[..., None]).any(-1).all(1).sum()
        )
        counts += int((predicted == label).sum())
        total += len(label)
        nll += float(
            F.cross_entropy(logits.reshape(-1, 257), label.reshape(-1), reduction="sum")
        )
        predicted_mass = (predicted * masses).sum(-1)
        target_mass = (label * masses).sum(-1)
        error += float((predicted_mass - target_mass).abs().sum())
    return {
        "molecules": total,
        "formula_top1": correct / total,
        "all_element_counts_in_top5": top5 / total,
        "element_accuracy": counts / (total * len(ELEMENTS)),
        "mean_count_nll": nll / (total * len(ELEMENTS)),
        "mean_argmax_mass_error_da": error / total,
        "selection_scope": "Formula diagnostic; cannot establish structure ranking gain",
    }


def train(output, epochs=12, seconds=3600, mass_weight=0.1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure(42, threads=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ROOT / "generation/smiles_42/model.pt"
    spec = {
        "version": 1,
        "epochs": epochs,
        "seconds": seconds,
        "mass_weight": mass_weight,
        "initial_checkpoint_sha256": digest(checkpoint),
        "train_sha256": digest(TRAIN),
        "dev_sha256": digest(ROOT / "researchdev.parquet"),
        "decoder_frozen": True,
        "loss": "element cross entropy plus Huber expected-neutral-mass error/100Da",
        "selection": "highest development exact formula; count NLL breaks ties",
        "oracle_formula_in_generation": False,
    }
    freeze(output / "protocol.json", spec)
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = FormulaPredictor(saved["condition_dim"]).to(device)
    model.load_state_dict(saved["formula"])
    train_frame = pd.read_parquet(TRAIN, columns=["inchikey14", "molecular_formula"])
    train_x = np.load(conditions(ROOT, "train60k") / "condition.npy", mmap_mode="r")
    train_labels = [formula_counts(f) for f in train_frame.molecular_formula]
    ids = np.array([i for i, y in enumerate(train_labels) if y is not None])
    labels = np.zeros((len(train_labels), len(ELEMENTS)), np.int64)
    labels[ids] = np.stack([train_labels[i] for i in ids])
    groups = train_frame.iloc[ids].groupby("inchikey14", sort=True).groups
    dev_frame = pd.read_parquet(
        ROOT / "researchdev.parquet", columns=["inchikey14", "molecular_formula"]
    )
    dev_x = np.load(conditions(ROOT, "researchdev") / "condition.npy", mmap_mode="r")
    dev_ids = [
        int(v[0]) for v in dev_frame.groupby("inchikey14", sort=True).indices.values()
    ]
    dev_ids = [
        i
        for i in dev_ids
        if formula_counts(dev_frame.molecular_formula.iloc[i]) is not None
    ]
    dev_y = np.stack(
        [formula_counts(dev_frame.molecular_formula.iloc[i]) for i in dev_ids]
    )
    dev_values = np.array(dev_x[dev_ids])
    masses = torch.tensor([ATOMIC_MASS[e] for e in ELEMENTS], device=device)
    baseline = evaluate(model, dev_values, dev_y, masses, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    budget = StageBudget(output, "generation", str(output), seconds)
    start = time.monotonic()
    best = (-1.0, float("-inf"))
    history = []
    rng = np.random.default_rng(42)
    try:
        for epoch in range(1, epochs + 1):
            if not budget.checkpoint():
                break
            order = np.array([rng.choice(np.array(v)) for v in groups.values()])
            rng.shuffle(order)
            model.train()
            summed, total = 0.0, 0
            for offset in range(0, len(order), 256):
                if not budget.checkpoint():
                    break
                batch = order[offset : offset + 256]
                x = torch.from_numpy(np.array(train_x[batch])).to(device)
                y = torch.from_numpy(labels[batch]).to(device)
                logits = model(x)
                count_loss = F.cross_entropy(logits.reshape(-1, 257), y.reshape(-1))
                expected = (logits.softmax(-1) * model.counts).sum(-1)
                mass_loss = F.smooth_l1_loss(
                    (expected * masses).sum(-1) / 100.0, (y * masses).sum(-1) / 100.0
                )
                loss = count_loss + mass_weight * mass_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                summed += float(loss.detach()) * len(batch)
                total += len(batch)
            report = evaluate(model, dev_values, dev_y, masses, device)
            history.append(
                {
                    "epoch": epoch,
                    "loss": summed / max(total, 1),
                    "development": report,
                    "seconds": time.monotonic() - start,
                }
            )
            print(history[-1], flush=True)
            score = (report["formula_top1"], -report["mean_count_nll"])
            if score > best:
                best = score
                improved = dict(saved)
                improved["formula"] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                improved["formula_training"] = spec
                torch.save(improved, output / "model.pt")
            write_json(output / "history.json", history)
        result = {
            "baseline": baseline,
            "selected": max(
                history,
                key=lambda h: (
                    h["development"]["formula_top1"],
                    -h["development"]["mean_count_nll"],
                ),
            )
            if history
            else None,
            "epochs": len(history),
            "seconds": time.monotonic() - start,
            "device": str(device),
            "decoder_frozen": True,
            "independent_acceptance": False,
            "next_gate": "Full 2000-molecule structure ranking required before release",
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seconds", type=float, default=3600)
    p.add_argument("--mass-weight", type=float, default=0.1)
    a = p.parse_args()
    print(json.dumps(train(a.output, a.epochs, a.seconds, a.mass_weight), indent=2))


if __name__ == "__main__":
    main()
