"""Matched encoder finetuning with fingerprint anchors and frozen near-mass critic."""

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
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

FEATURES = Path("artifacts/scale_20260929/features/train60k")
HARD = Path(
    "artifacts/research_loop/rounds/0084_critic_hard_negatives/hard_negatives.npy"
)


def validate_training_binding(data, training, cached, development_keys, saved):
    if not cached.reset_index(drop=True).equals(training.reset_index(drop=True)):
        raise ValueError("Training feature row mapping differs")
    groups = training.groupby("inchikey14", sort=True).indices
    if (
        len(data["keys"]) != 60000
        or len(set(data["keys"])) != 60000
        or set(data["keys"]) != set(groups)
        or set(groups) & development_keys
    ):
        raise ValueError("Training/development scope isolation failed")
    if (
        any(
            not np.array_equal(ids, groups[key])
            for key, ids in zip(data["keys"], data["rows"])
        )
        or len(data["rows"]) != 60000
    ):
        raise ValueError("Direct training rows differ from encoder feature mapping")
    features_manifest = json.loads((FEATURES / "complete.json").read_text())
    if features_manifest["preprocessing"] != saved[
        "preprocessing"
    ] or features_manifest["rows"] != len(training):
        raise ValueError("Cached features/encoder preprocessing differ")
    direct_manifest = json.loads((DIRECT / "training_manifest.json").read_text())
    hard_protocol = json.loads((HARD.parent / "protocol.json").read_text())
    expected = {
        "encoder_sha256": digest(ENCODER),
        "critic_sha256": digest(CRITIC),
        "training_sha256": digest(TRAIN),
        "training_data_sha256": digest(DIRECT / "training.joblib"),
    }
    if direct_manifest["encoder_sha256"] != expected["encoder_sha256"] or any(
        hard_protocol.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Hard-negative mining source binding differs")


def validate_hard_table(hard, data):
    count = len(data["keys"])
    if (
        hard.shape != (count, 15)
        or not np.issubdtype(hard.dtype, np.integer)
        or (hard < 0).any()
        or (hard >= count).any()
        or (hard == np.arange(count)[:, None]).any()
        or any(len(set(row)) != 15 for row in hard)
    ):
        raise ValueError("Invalid training-only hard table")
    if any(
        not set(hard[i]).issubset(
            data["negatives"][i, : max(16, int(data["inside"][i]))]
        )
        for i in range(count)
    ):
        raise ValueError("Hard negatives outside frozen training mass pool")


def freeze_protocol(output, contrastive=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    return freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "encoder_sha256": digest(ENCODER),
            "critic_sha256": digest(CRITIC),
            "training_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "training_data_sha256": digest(DIRECT / "training.joblib"),
            "features_complete_sha256": digest(FEATURES / "complete.json"),
            "hard_negatives_sha256": digest(HARD),
            "hard_mining_protocol_sha256": digest(HARD.parent / "protocol.json"),
            "features_sha256": {
                name: digest(FEATURES / f"{name}.npy")
                for name in ("hist", "meta", "loss")
            },
            "contrastive": contrastive,
            "loss": "Fingerprint BCE plus0.1 training-only15 hard near-mass negative listwise CE, frozen original critic",
            "contrastive_weight": 0.1 if contrastive else 0.0,
            "epochs": 3,
            "batch_size": 128,
            "learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "seed": 42,
            "matched_control": "Same initial encoder/critic, one equally weighted spectrum per training molecule per epoch, same hard table, positive/permutation/dropout RNG and both forwards",
            "critic_frozen": True,
            "encoder_trainable": True,
            "head_fingerprint_anchor": True,
            "selection": "Fixed epoch3; development labels not used for training or checkpoint selection",
            "scope": "External candidate scoring encoder only; incumbent retrieval/generation remain frozen0062",
            "representation_seconds": 86400,
            "adversarial_training": False,
            "cohort": "repeated_development",
            "holdout_used": False,
        },
    )


def run(output, contrastive=False):
    output = Path(output)
    freeze_protocol(output, contrastive)
    configure(42, threads=4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for matched encoder control")
    budget = StageBudget(output, "representation", "encoder_fixed3epochs", 86400)
    started = time.monotonic()
    try:
        data = train_data()
        training = pd.read_parquet(TRAIN, columns=["inchikey14", "row_id"])
        cached = pd.read_parquet(FEATURES / "rows.parquet")
        devkeys = set(
            pd.read_parquet(
                ROOT / "researchdev.parquet", columns=["inchikey14"]
            ).inchikey14
        )
        device = torch.device("cuda")
        model, saved = load_deployment_checkpoint(ENCODER, "scale")
        validate_training_binding(data, training, cached, devkeys, saved)
        hard = np.load(HARD)
        validate_hard_table(hard, data)
        model.to(device)
        weights = torch.load(CRITIC, map_location="cpu", weights_only=True)
        if (
            weights["encoder_sha256"] != digest(ENCODER)
            or weights["architecture"] != "fingerprint"
        ):
            raise ValueError("Initial critic/encoder binding failed")
        critic = DirectRanker("fingerprint").to(device).eval().requires_grad_(False)
        critic.load_state_dict(weights["state_dict"])
        arrays = {
            name: np.load(FEATURES / f"{name}.npy", mmap_mode="r")
            for name in ("hist", "meta", "loss")
        }
        expected_columns = {"hist": 1250, "meta": saved["metadata_dim"], "loss": 1250}
        if (
            any(
                arrays[name].shape != (len(training), columns)
                or arrays[name].dtype != np.float32
                for name, columns in expected_columns.items()
            )
            or data["fps"].shape != (60000, 2048)
            or data["fps"].dtype != np.float32
        ):
            raise ValueError("Training feature/fingerprint dimensions differ")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        rng = np.random.default_rng(42)
        history = []
        for epoch in range(1, 4):
            model.train()
            bces, pairs = [], []
            for ids in np.array_split(
                rng.permutation(60000), int(np.ceil(60000 / 128))
            ):
                if not budget.checkpoint():
                    raise TimeoutError("Encoder finetuning budget exhausted")
                positive = np.asarray([rng.choice(data["rows"][i]) for i in ids])
                chosen = np.column_stack([ids, hard[ids]])
                unique, inverse = np.unique(chosen, return_inverse=True)
                features = torch.cat(
                    [
                        torch.from_numpy(np.array(arrays[name][positive])).to(device)
                        for name in ("hist", "meta", "loss")
                    ],
                    -1,
                )
                fps = torch.from_numpy(data["fps"][unique]).to(device)
                target = torch.from_numpy(data["fps"][ids]).to(device)
                optimizer.zero_grad(set_to_none=True)
                z = model.encoder(features)
                logits = critic(
                    z,
                    fps,
                    None,
                    torch.from_numpy(inverse.reshape(chosen.shape)).to(device),
                )
                anchor = F.binary_cross_entropy_with_logits(model.head(z), target)
                pair = F.cross_entropy(
                    logits, torch.zeros(len(ids), dtype=torch.long, device=device)
                )
                loss = anchor + (0.1 if contrastive else 0.0) * pair
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite encoder anchored contrastive loss"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                bces.append(float(anchor.detach()))
                pairs.append(float(pair.detach()))
            row = {
                "epoch": epoch,
                "fingerprint_bce": float(np.mean(bces)),
                "frozen_critic_pair_ce": float(np.mean(pairs)),
                "seconds": time.monotonic() - started,
            }
            history.append(row)
            write_json(output / "history.json", history)
            print(json.dumps(row), flush=True)
        if any(
            not torch.equal(value.detach().cpu(), weights["state_dict"][key])
            for key, value in critic.state_dict().items()
        ):
            raise ValueError("Frozen critic weights changed")
        if any(
            not torch.isfinite(value).all() for value in model.state_dict().values()
        ):
            raise FloatingPointError("Nonfinite final encoder weights")
        saved["state_dict"] = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        saved["finetuning_protocol"] = json.loads(
            (output / "protocol.json").read_text()
        )
        saved["epoch"] = 3
        torch.save(saved, output / "model.pt")
        _, reloaded = load_deployment_checkpoint(output / "model.pt", "scale")
        if reloaded["preprocessing"] != saved["preprocessing"]:
            raise ValueError("Final encoder preprocessing changed")
        # Same critic weights; updated binding describes the new scoring encoder.
        weights["encoder_sha256"] = digest(output / "model.pt")
        weights["original_critic_sha256"] = digest(CRITIC)
        torch.save(weights, output / "critic.pt")
        result = {
            "diagnostic_only": True,
            "contrastive": contrastive,
            "train_molecules": 60000,
            "train_spectra": len(training),
            "train_development_overlap": 0,
            "epochs": 3,
            "critic_state_dict_unchanged": True,
            "seconds": time.monotonic() - started,
            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            "encoder_sha256": digest(output / "model.pt"),
            "critic_sha256": digest(output / "critic.pt"),
            "history": history,
            "independent_acceptance": False,
            "next_gate": "Complete external and combined scoring against0062 required; training loss cannot qualify release",
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrastive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.contrastive), indent=2))


if __name__ == "__main__":
    main()
