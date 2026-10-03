"""Fit a bounded candidate-specific chemical bonus using training-only pinned negatives."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from casmi_ml.candidate_diagnostic_compatibility import BONUS_BUDGET, structural_support
from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import DIM, evidence_features
from casmi_ml.direct_experiment import train_data
from casmi_ml.direct_models import DirectRanker
from casmi_ml.encoder_pair_finetuning import (
    FEATURES,
    HARD,
    validate_hard_table,
    validate_training_binding,
)
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure
from casmi_ml.training_query_pairs import paired_rows


def fit_bonus(cosines, interactions, scale, mask=False, checkpoint=None):
    cosines = np.asarray(cosines, np.float64)
    interactions = np.asarray(interactions, np.float64)
    if (
        cosines.ndim != 2
        or cosines.shape[1] != 16
        or interactions.shape != (*cosines.shape, DIM)
    ):
        raise ValueError(
            "Positive+15 candidate scores/chemical24 interactions must align"
        )
    if (
        not np.isfinite(cosines).all()
        or not np.isfinite(interactions).all()
        or not (0 < scale <= 50)
    ):
        raise ValueError("Finite frozen critic and chemistry required")
    if ((interactions < 0) | (interactions > 1)).any():
        raise ValueError("Candidate interactions must lie within [0,1]")
    if mask:
        interactions = np.zeros_like(interactions)

    def objective(weight):
        if checkpoint is not None and not checkpoint():
            raise RuntimeError("Candidate chemistry stage budget exhausted")
        bonus = interactions @ weight
        logits = (cosines + bonus) * scale
        normalizer = logsumexp(logits, axis=1)
        probability = np.exp(logits - normalizer[:, None])
        probability[:, 0] -= 1
        loss = (normalizer - logits[:, 0]).mean() + 0.01 * (weight @ weight) / 2
        gradient = (
            np.einsum("bc,bcd->d", probability, interactions) * scale / len(cosines)
            + 0.01 * weight
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(DIM),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, BONUS_BUDGET)] * DIM,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda w: BONUS_BUDGET - w.sum(),
                "jac": lambda w: -np.ones(DIM),
            }
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(
            "Bounded candidate chemistry optimizer failed: " + result.message
        )
    weight = np.maximum(result.x, 0)
    if weight.sum() > BONUS_BUDGET:
        weight *= BONUS_BUDGET / weight.sum()
    return weight, {
        "initial_ce": float(objective(np.zeros(DIM))[0]),
        "final_objective": float(objective(weight)[0]),
        "optimizer_iterations": int(result.nit),
        "sum_nonnegative_cosine_weights": float(weight.sum()),
        "maximum_possible_bonus": BONUS_BUDGET,
    }


@torch.inference_mode()
def run(output, mask_chemical_channels=False):
    configure(42, threads=4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU needed for frozen training-score preparation")
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "compatibility_source_sha256": digest(
                "casmi_ml/candidate_diagnostic_compatibility.py"
            ),
            "chemistry_source_sha256": digest("casmi_ml/chemistry.py"),
            "evidence_source_sha256": digest("casmi_ml/diagnostic_residual.py"),
            "feature_source_sha256": digest("casmi_ml/diagnostic_feature_runtime.py"),
            "training_sha256": digest(TRAIN),
            "encoder_sha256": digest(ENCODER),
            "critic_sha256": digest(CRITIC),
            "hard_table_sha256": digest(HARD),
            "rule": "Frozen original critic cosine plus sum of nonnegative24 mean observed precise chemistry times actual candidate broad SMARTS support; weights sum<=existing admission margin.05, so each total candidate bonus<=.05. No absent-peak deduction or new structural claim.",
            "training": "Same60K originalRNG42 representative+same-ion pairRNG260, same15 unrestricted pinned negatives; original encoder/critic all frozen; convex24D SLSQP bounds/sum budget, fixedL2.01. All training scores once, no dev parameter selection.",
            "mask_chemical_channels": mask_chemical_channels,
            "representation_seconds": 86400,
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    budget = StageBudget(
        output,
        "representation",
        "candidate_compatibility_fixed",
        86400,
        lock_path=output / "representation.lock",
    )
    try:
        return prepare_and_fit(output, mask_chemical_channels, budget)
    finally:
        budget.close()


@torch.inference_mode()
def prepare_and_fit(output, mask_chemical_channels, budget):
    started = time.monotonic()

    def checkpoint():
        if not budget.checkpoint():
            raise RuntimeError("Candidate chemistry stage budget exhausted")

    checkpoint()
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
    groups = train.groupby("inchikey14", sort=True).indices
    rng = np.random.default_rng(42)
    ids = np.array([rng.choice(v) for v in groups.values()])
    paired = paired_rows(train, ids)
    selected_rows = np.column_stack([ids, paired]).reshape(-1)
    support = np.asarray(
        [structural_support(s) for s in train.iloc[ids].normalized_smiles], np.float32
    )
    chemistry_rows = []
    for start in range(0, len(selected_rows), 1024):
        checkpoint()
        chemistry_rows.extend(
            evidence_features(r, True)
            for r in train.iloc[selected_rows[start : start + 1024]].to_dict("records")
        )
    chemical = np.stack(chemistry_rows).reshape(60000, 2, DIM).mean(1)
    del chemistry_rows
    # Prepared pairs use the same sorted molecule identities as the pinned table.
    if list(groups) != list(data["keys"]):
        raise ValueError(
            "Training candidate SMARTS identities differ from pinned order"
        )
    encoder.eval().requires_grad_(False).to("cuda")
    weights = torch.load(CRITIC, map_location="cpu", weights_only=True)
    if weights["encoder_sha256"] != digest(ENCODER):
        raise ValueError("Frozen original encoder binding differs")
    critic = DirectRanker("fingerprint").eval().requires_grad_(False).to("cuda")
    critic.load_state_dict(weights["state_dict"])
    arrays = {
        n: np.load(FEATURES / (n + ".npy"), mmap_mode="r")
        for n in ["hist", "meta", "loss"]
    }
    queries = torch.empty(len(selected_rows), 128, device="cuda")
    for start in range(0, len(selected_rows), 256):
        checkpoint()
        features = torch.cat(
            [
                torch.from_numpy(np.array(a[selected_rows[start : start + 256]])).to(
                    "cuda"
                )
                for a in arrays.values()
            ],
            -1,
        )
        queries[start : start + 256] = critic.encode_spectra(encoder.encoder(features))
    queries = queries.reshape(60000, 2, 128).mean(1)
    molecules = torch.cat(
        [
            critic.encode_molecules(
                torch.from_numpy(data["fps"][start : start + 256]).to("cuda")
            )
            for start in range(0, 60000, 256)
        ]
    )
    selected = np.concatenate([np.arange(60000)[:, None], hard], axis=1)
    cosines = np.concatenate(
        [
            (
                queries[start:stop, None]
                * molecules[torch.from_numpy(selected[start:stop]).to("cuda")]
            )
            .sum(-1)
            .cpu()
            .numpy()
            for start in range(0, 60000, 256)
            for stop in [min(start + 256, 60000)]
        ]
    )
    interactions = chemical[:, None, :] * np.repeat(support[selected], 3, axis=-1)
    scale = float(critic.log_scale.exp().clamp(max=50))
    bonus, history = fit_bonus(
        cosines, interactions, scale, mask_chemical_channels, budget.checkpoint
    )
    model = {
        "weights": bonus.tolist(),
        "mask_chemical_channels": mask_chemical_channels,
        "bonus_budget": BONUS_BUDGET,
        "original_critic_sha256": digest(CRITIC),
        "encoder_sha256": digest(ENCODER),
        "compatibility_source_sha256": digest(
            "casmi_ml/candidate_diagnostic_compatibility.py"
        ),
        "chemistry_source_sha256": digest("casmi_ml/chemistry.py"),
        "evidence_source_sha256": digest("casmi_ml/diagnostic_residual.py"),
        "feature_source_sha256": digest("casmi_ml/diagnostic_feature_runtime.py"),
        "training_sha256": digest(TRAIN),
        "training_molecules": 60000,
        "training_spectra_pair_slots": 120000,
        "objective": history,
        "local_only": True,
        "independent_acceptance": False,
    }
    write_json(output / "model.json", model)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "train_development_overlap": 0,
        "original_encoder_and_critic_frozen": True,
        "history": history,
        "seconds": time.monotonic() - started,
        "model_sha256": digest(output / "model.json"),
        "scope": "Bounded candidate chemistry fitting only; current-best complete zero control and paired full3340 systems required",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--mask-chemical-channels", action="store_true")
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
