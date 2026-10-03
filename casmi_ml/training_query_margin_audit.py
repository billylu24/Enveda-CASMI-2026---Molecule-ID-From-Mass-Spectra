"""Training-only frozen query gaps versus the deployed candidate admission margin."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import write_json
from casmi_ml.diagnostic_query_ranker import load_query_ranker
from casmi_ml.diagnostic_residual import DIM, evidence_features
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import (
    FEATURES,
    HARD,
    validate_hard_table,
    validate_training_binding,
)
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure
from casmi_ml.training_query_pairs import paired_rows

MODELS = {
    "paired_contextual": "0260_query_multispectrum_chemical",
    "chemical_only": "0266_query_chemical_only",
}


def gap_summary(positive, negatives):
    gap = positive[:, None] - negatives
    hardest = gap.min(1)
    return {
        "molecules": len(positive),
        "positive_above_all_pinned_negatives": int((hardest > 0).sum()),
        "positive_above_all_by_deployed_margin": int((hardest > 0.05).sum()),
        "rank_correct_but_gap_at_most_deployed_margin": int(
            ((hardest > 0) & (hardest <= 0.05)).sum()
        ),
        "hardest_gap_quantiles": np.quantile(hardest, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "per_negative_margin_violations": int((gap <= 0.05).sum()),
    }


@torch.inference_mode()
def run(output):
    configure(42, threads=4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for training-only margin diagnostic")
    output = Path(output)
    paths = {
        name: Path("artifacts/research_loop/rounds") / rid / "model.pt"
        for name, rid in MODELS.items()
    }
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "encoder_sha256": digest(ENCODER),
            "critic_sha256": digest(CRITIC),
            "hard_table_sha256": digest(HARD),
            "models_sha256": {name: digest(path) for name, path in paths.items()},
            "scope": "All60K training-only paired queries, pinned15 negatives; cosine scores before scale exactly as admission gate, no development scores or fitting",
            "sampling": "OriginalRNG42 representative+same-ion measured pairRNG260, each molecule equally weighted",
            "deployed_margin": 0.05,
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    started = time.monotonic()
    train = pd.read_parquet(TRAIN)
    data = train_data()
    encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
    devkeys = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    validate_training_binding(
        data,
        train[["inchikey14", "row_id"]].reset_index(drop=True),
        pd.read_parquet(FEATURES / "rows.parquet"),
        devkeys,
        saved,
    )
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    rng = np.random.default_rng(42)
    ids = np.array(
        [rng.choice(v) for v in train.groupby("inchikey14", sort=True).indices.values()]
    )
    paired = paired_rows(train, ids)
    positions = np.column_stack([ids, paired]).reshape(-1)
    chemical = (
        torch.from_numpy(
            np.stack(
                [
                    evidence_features(row, True)
                    for row in train.iloc[positions].to_dict("records")
                ]
            )
        )
        .to("cuda")
        .reshape(60000, 2, DIM)
    )
    encoder.eval().requires_grad_(False).to("cuda")
    arrays = {
        n: np.load(FEATURES / (n + ".npy"), mmap_mode="r")
        for n in ["hist", "meta", "loss"]
    }
    latent = torch.empty(len(positions), 768, device="cuda")
    for start in range(0, len(positions), 256):
        features = torch.cat(
            [
                torch.from_numpy(np.array(a[positions[start : start + 256]])).to("cuda")
                for a in arrays.values()
            ],
            -1,
        )
        latent[start : start + 256] = encoder.encoder(features)
    latent = latent.reshape(60000, 2, 768)
    models = {
        name: load_query_ranker(path, CRITIC)[0].to("cuda")
        for name, path in paths.items()
    }
    base = next(iter(models.values())).base
    molecules = torch.cat(
        [
            base.encode_molecules(
                torch.from_numpy(data["fps"][start : start + 256]).to("cuda")
            )
            for start in range(0, 60000, 256)
        ]
    )
    hard_tensor = torch.from_numpy(hard).to("cuda")
    arms = {name: [] for name in ["frozen_baseline", *models]}
    for start in range(0, 60000, 256):
        stop = min(start + 256, 60000)
        contexts = latent[start:stop].reshape(-1, 768)
        chem = chemical[start:stop].reshape(-1, DIM)
        queries = {
            "frozen_baseline": base.encode_spectra(contexts).reshape(-1, 2, 128).mean(1)
        }
        queries.update(
            {
                name: model.encode_spectra(contexts, chem).reshape(-1, 2, 128).mean(1)
                for name, model in models.items()
            }
        )
        indexes = torch.arange(start, stop, device="cuda")
        selected = torch.cat([indexes[:, None], hard_tensor[start:stop]], -1)
        structures = molecules[selected]
        for name, query in queries.items():
            arms[name].append((query[:, None] * structures).sum(-1).cpu().numpy())
    summaries = {}
    for name, parts in arms.items():
        values = np.concatenate(parts)
        summaries[name] = gap_summary(values[:, 0], values[:, 1:])
    packed = np.packbits(data["fps"].astype(np.uint8), axis=1)
    identical = sum(
        int((packed[start:stop, None] == packed[hard[start:stop]]).all(-1).sum())
        for start in range(0, 60000, 256)
        for stop in [min(start + 256, 60000)]
    )
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "train_development_overlap": 0,
        "arms": summaries,
        "identical_positive_negative_fingerprint_pairs": identical,
        "seconds": time.monotonic() - started,
        "scope": "Cosine-margin training diagnostic; no development checkpoint selection or full-system gain claim",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
