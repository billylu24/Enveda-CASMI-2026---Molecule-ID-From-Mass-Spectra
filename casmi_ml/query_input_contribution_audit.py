"""Training-only audit of latent and chemistry contributions to frozen query heads."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml.data import write_json
from casmi_ml.diagnostic_query_ranker import load_query_ranker
from casmi_ml.diagnostic_residual import evidence_features
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import FEATURES, validate_training_binding
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure
from casmi_ml.training_query_pairs import paired_rows

MODELS = {
    "single_chemical": "0245_diagnostic_query_chemical",
    "single_masked": "0246_diagnostic_query_mask_control",
    "paired_chemical": "0260_query_multispectrum_chemical",
    "paired_masked": "0261_query_multispectrum_mask_control",
}


def rms(value):
    return float(value.square().mean().sqrt())


@torch.inference_mode()
def run(output):
    configure(42, threads=4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        k: Path("artifacts/research_loop/rounds") / v / "model.pt"
        for k, v in MODELS.items()
    }
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "encoder_sha256": digest(ENCODER),
            "critic_sha256": digest(CRITIC),
            "models_sha256": {k: digest(v) for k, v in paths.items()},
            "selection": "4096 evenly spaced original train-key positions; RNG42 representatives and original RNG260 same-ion second rows, no development outcome selection",
            "purpose": "First linear latent/chemical contribution and query identity counterfactual; descriptive training-only audit, not accuracy evidence",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    started = time.monotonic()
    train = pd.read_parquet(TRAIN)
    encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
    encoder.eval().requires_grad_(False)
    devkeys = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    validate_training_binding(
        train_data(),
        train[["inchikey14", "row_id"]].reset_index(drop=True),
        pd.read_parquet(FEATURES / "rows.parquet"),
        devkeys,
        saved,
    )
    rng = np.random.default_rng(42)
    ids = np.array(
        [rng.choice(v) for v in train.groupby("inchikey14", sort=True).indices.values()]
    )
    paired = paired_rows(train, ids)
    positions = np.linspace(0, len(ids) - 1, 4096, dtype=int)
    selected = np.column_stack([ids[positions], paired[positions]]).reshape(-1)
    chemical = torch.from_numpy(
        np.stack(
            [
                evidence_features(r, True)
                for r in train.iloc[selected].to_dict("records")
            ]
        )
    )
    arrays = {
        name: np.load(FEATURES / (name + ".npy"), mmap_mode="r")
        for name in ["hist", "meta", "loss"]
    }
    latent = torch.cat(
        [
            encoder.encoder(
                torch.cat(
                    [
                        torch.from_numpy(np.array(a[selected[start : start + 256]]))
                        for a in arrays.values()
                    ],
                    -1,
                )
            )
            for start in range(0, len(selected), 256)
        ]
    )
    arms = {}
    for name, path in paths.items():
        ranker, _ = load_query_ranker(path, CRITIC)
        head = ranker.query_residual
        layer = head.correction[0]
        identity = (
            torch.zeros_like(chemical) if head.mask_chemical_channels else chemical
        )
        context_term = F.linear(latent, layer.weight[:, :768])
        chemical_term = F.linear(identity, layer.weight[:, 768:])
        baseline = ranker.base.encode_spectra(latent)
        corrected = head(baseline, latent, chemical)
        hidden_masked = F.gelu(F.linear(latent, layer.weight[:, :768], layer.bias))
        delta_masked = head.correction[-1](hidden_masked) * chemical.amax(
            -1, keepdim=True
        )
        # Parenthesize normalization difference to match actual residual arithmetic.
        masked = baseline + (
            F.normalize(baseline + delta_masked, dim=-1) - F.normalize(baseline, dim=-1)
        )
        arms[name] = {
            "latent_linear_rms": rms(context_term),
            "chemical_linear_rms": rms(chemical_term),
            "chemical_over_latent_linear_rms": rms(chemical_term)
            / max(rms(context_term), 1e-12),
            "corrected_query_displacement_mean_l2": float(
                (corrected - baseline).norm(dim=-1).mean()
            ),
            "chemical_identity_counterfactual_mean_l2": float(
                (corrected - masked).norm(dim=-1).mean()
            ),
            "chemical_identity_counterfactual_max_l2": float(
                (corrected - masked).norm(dim=-1).max()
            ),
            "evidence_rows": int(chemical.amax(-1).gt(0).sum()),
            "rows": len(selected),
        }
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": len(positions),
        "training_rows": len(selected),
        "arms": arms,
        "seconds": time.monotonic() - started,
        "scope": "Training-only descriptive mechanism audit; no development rank evaluation or acceptance claim",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
