"""Matched short finetuning: training-only random vs frozen-critic hard negatives."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml.data import write_json
from casmi_ml.direct_experiment import ROOT as DIRECT
from casmi_ml.direct_experiment import train_data
from casmi_ml.direct_models import DirectRanker
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.training import configure


def choose_hard(available, scores, keys, count=15):
    if len(available) != len(scores) or len(set(available)) != len(available):
        raise ValueError("Unique scored negative candidates required")
    if len(available) < count or not np.isfinite(scores).all():
        raise ValueError("Insufficient or nonfinite negative scores")
    score_map = dict(zip(available, scores))
    return np.asarray(
        sorted(available, key=lambda i: (-float(score_map[i]), keys[i]))[:count],
        dtype=np.int64,
    )


def run(output, hard=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latent_path = DIRECT / "encoder_train60k.npz"
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "training_sha256": digest(TRAIN),
            "critic_sha256": digest(CRITIC),
            "encoder_sha256": digest(ENCODER),
            "latent_cache_sha256": digest(latent_path),
            "training_data_sha256": digest(DIRECT / "training.joblib"),
            "hard_negatives": hard,
            "epochs": 3,
            "learning_rate": 1e-5,
            "batch_size": 128,
            "seed": 42,
            "negative_count": 15,
            "negative_pool": "Original training-only nearest mass128; pool width max16/in-window count",
            "mining": "Frozen initial critic on mean normalized training-spectrum embeddings; deterministic key ties",
            "matched_rng": "Both arms consume identical random negative draws and positive/permutation RNG; hard arm replaces only negatives",
            "checkpoint_selection": "Fixed third epoch, no development-based early stopping",
            "encoder_frozen": True,
            "development_labels_used_in_training": False,
            "holdout_used": False,
            "adversarial_training": False,
            "representation_seconds": 86400,
        },
    )
    configure(42, threads=4)
    budget = StageBudget(output, "representation", "critic_finetuning", 86400)
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for matched critic training")
        data = train_data()
        training_frame = pd.read_parquet(TRAIN, columns=["inchikey14"])
        training_groups = training_frame.groupby("inchikey14", sort=True).indices
        training_keys = set(training_groups)
        dev_keys = set(
            pd.read_parquet(
                ROOT / "researchdev.parquet", columns=["inchikey14"]
            ).inchikey14
        )
        if (
            len(data["keys"]) != 60000
            or set(data["keys"]) != training_keys
            or training_keys & dev_keys
        ):
            raise ValueError("Training/development isolation failed")
        if any(
            not np.array_equal(ids, training_groups[key])
            for key, ids in zip(data["keys"], data["rows"])
        ):
            raise ValueError("Training latent row mapping changed")
        manifest = json.loads((DIRECT / "training_manifest.json").read_text())
        weights = torch.load(CRITIC, map_location="cpu", weights_only=True)
        if weights["architecture"] != "fingerprint":
            raise ValueError("Require fingerprint critic")
        if manifest["encoder_sha256"] != digest(ENCODER) or weights[
            "encoder_sha256"
        ] != digest(ENCODER):
            raise ValueError("Latent/critic encoder binding failed")
        device = torch.device("cuda")
        model = DirectRanker("fingerprint").to(device).eval()
        model.load_state_dict(weights["state_dict"])
        latent = np.load(latent_path)["latent"]
        if latent.shape != (sum(len(r) for r in data["rows"]), 768):
            raise ValueError("Training cache shape mismatch")
        # Both arms compute the same frozen mining table; only usage differs.
        with torch.inference_mode():
            structure = torch.cat(
                [
                    model.encode_molecules(
                        torch.from_numpy(data["fps"][i : i + 512]).to(device)
                    )
                    for i in range(0, len(data["keys"]), 512)
                ]
            )
            query_rows = np.empty((len(data["keys"]), 128), np.float32)
            for i in range(0, len(latent), 1024):
                # Store normalized individual spectra, then aggregate per molecule.
                if i == 0:
                    all_query = np.empty((len(latent), 128), np.float32)
                all_query[i : i + 1024] = (
                    model.encode_spectra(
                        torch.from_numpy(latent[i : i + 1024]).to(device)
                    )
                    .cpu()
                    .numpy()
                )
            for i, ids in enumerate(data["rows"]):
                query_rows[i] = all_query[ids].mean(0)
            del all_query
            hard_table = np.empty((len(data["keys"]), 15), np.int64)
            for start in range(0, len(data["keys"]), 256):
                end = min(start + 256, len(data["keys"]))
                neighbors = data["negatives"][start:end]
                scores = (
                    (
                        structure[torch.from_numpy(neighbors).to(device)]
                        * torch.from_numpy(query_rows[start:end]).to(device)[:, None]
                    )
                    .sum(-1)
                    .cpu()
                    .numpy()
                )
                for local, i in enumerate(range(start, end)):
                    width = max(16, int(data["inside"][i]))
                    available = neighbors[local, :width].tolist()
                    if i in available:
                        raise ValueError("Positive included as a negative")
                    hard_table[i] = choose_hard(
                        available, scores[local, :width], data["keys"]
                    )
                if not budget.checkpoint():
                    raise TimeoutError("Critic mining budget exhausted")
            del structure, query_rows
        np.save(output / "hard_negatives.npy", hard_table)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        rng = np.random.default_rng(42)
        history = []
        for epoch in range(1, 4):
            model.train()
            losses, pairs_inside, pairs_total = [], 0, 0
            for ids in np.array_split(
                rng.permutation(60000), int(np.ceil(60000 / 128))
            ):
                positive = np.array([rng.choice(data["rows"][i]) for i in ids])
                random_negative = np.stack(
                    [
                        rng.choice(
                            data["negatives"][i, : max(16, int(data["inside"][i]))],
                            15,
                            replace=False,
                        )
                        for i in ids
                    ]
                )
                negative = hard_table[ids] if hard else random_negative
                chosen = np.column_stack([ids, negative])
                unique, inverse = np.unique(chosen, return_inverse=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    torch.from_numpy(latent[positive]).to(device),
                    torch.from_numpy(data["fps"][unique]).to(device),
                    None,
                    torch.from_numpy(inverse.reshape(chosen.shape)).to(device),
                )
                loss = F.cross_entropy(
                    logits, torch.zeros(len(ids), dtype=torch.long, device=device)
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite critic training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                tolerance = np.maximum(0.006, data["masses"][ids] * 35e-6)
                pairs_inside += int(
                    (
                        np.abs(data["masses"][negative] - data["masses"][ids, None])
                        <= tolerance[:, None]
                    ).sum()
                )
                pairs_total += int(negative.size)
                if not budget.checkpoint():
                    raise TimeoutError("Critic training budget exhausted")
            entry = {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                "pairs": pairs_total,
                "pairs_inside_mass_window": pairs_inside,
                "seconds": time.monotonic() - started,
            }
            history.append(entry)
            write_json(output / "history.json", history)
            print(entry, flush=True)
        report = {
            "diagnostic_only": True,
            "training_molecules": 60000,
            "training_development_overlap": 0,
            "hard_negatives": hard,
            "history": history,
            "seconds": time.monotonic() - started,
            "independent_acceptance": False,
            "performance_evaluation_pending": True,
            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        torch.save(
            {
                "architecture": "fingerprint",
                "encoder_sha256": digest(ENCODER),
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "training_report": report,
            },
            output / "model.pt",
        )
        report["checkpoint_sha256"] = digest(output / "model.pt")
        write_json(output / "report.json", report)
        return report
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hard", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a.output, a.hard), indent=2))


if __name__ == "__main__":
    main()
