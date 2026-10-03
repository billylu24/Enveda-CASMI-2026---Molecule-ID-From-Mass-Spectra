"""Matched precise chemical versus masked query residuals on training-only hard negatives."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml.data import write_json
from casmi_ml.diagnostic_query_ranker import DiagnosticQueryRanker
from casmi_ml.diagnostic_residual import DIM, evidence_features
from casmi_ml.direct_experiment import ROOT as DIRECT
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
from casmi_ml.strict_mass_negative_mask import admissible_negatives
from casmi_ml.training import configure
from casmi_ml.training_query_pairs import paired_rows


def run(
    output,
    mask_chemical_channels=False,
    strict_mass_negatives=False,
    multispectrum=False,
):
    configure(42, threads=4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "query_residual_sha256": digest("casmi_ml/diagnostic_query_residual.py"),
            "query_ranker_sha256": digest("casmi_ml/diagnostic_query_ranker.py"),
            "training_sha256": digest(TRAIN),
            "encoder_sha256": digest(ENCODER),
            "critic_sha256": digest(CRITIC),
            "training_data_sha256": digest(DIRECT / "training.joblib"),
            "hard_table_sha256": digest(HARD),
            "precise": True,
            "mask_chemical_channels": mask_chemical_channels,
            "strict_mass_negatives": strict_mass_negatives,
            "multispectrum": multispectrum,
            "training_molecules": 60000,
            "row_selection": "one RNG42 representative per original60K molecule, same arms",
            "epochs": 3,
            "query_pair_source_sha256": digest("casmi_ml/training_query_pairs.py"),
            "query_aggregation": "Two measured rows under same molecule/adduct/mode, representativeRNG42 plus separate pairRNG260; per-spectrum corrected normalized queries averaged before CE and displacement anchor. Singleton row duplicated exactly."
            if multispectrum
            else "One originalRNG42 representative row per molecule",
            "optimizer": "AdamW .001 wd.01",
            "objective": (
                "Original frozen scale listwise CE over positive and admissible pinned negatives inside35ppm/floor.006Da; other logits excluded, all original60K steps retained + .1 squared query displacement anchor"
                if strict_mass_negatives
                else "Original frozen scale listwise CE over training positive and15 pinned near-mass hard negatives + .1 squared query displacement anchor"
            ),
            "strict_mass_source_sha256": digest(
                "casmi_ml/strict_mass_negative_mask.py"
            ),
            "architecture": "Frozen768 latent plus24 precise chemistry ->64 GELU ->128 zero-initialized query residual; same maximum chemistry strength gate and normalized original critic space; all original encoder and critic tensors frozen",
            "selection": "Fixed epoch3; development or holdout labels not used for training or checkpoint selection",
            "local_only": True,
            "independent_acceptance": False,
            "representation_seconds": 86400,
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    budget = StageBudget(
        output,
        "representation",
        "chemical_query_residual_fixed3",
        86400,
        lock_path=output / "representation.lock",
    )
    started = time.monotonic()
    try:
        train = pd.read_parquet(TRAIN)
        data = train_data()
        encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
        training = train[["inchikey14", "row_id"]].reset_index(drop=True)
        cached = pd.read_parquet(FEATURES / "rows.parquet")
        devkeys = set(
            pd.read_parquet(
                ROOT / "researchdev.parquet", columns=["inchikey14"]
            ).inchikey14
        )
        validate_training_binding(data, training, cached, devkeys, saved)
        hard = np.load(HARD)
        validate_hard_table(hard, data)
        strict_allowed = admissible_negatives(data["masses"], hard)
        strict_inside = int(strict_allowed.sum())
        strict_allowed = torch.from_numpy(strict_allowed).to("cuda")
        groups = train.groupby("inchikey14", sort=True).indices
        if list(groups) != list(data["keys"]):
            raise ValueError("Hard-negative and training key order differ")
        rng = np.random.default_rng(42)
        ids = np.array([rng.choice(v) for v in groups.values()])
        chosen = train.iloc[ids]
        target = np.stack(
            [
                np.unpackbits(np.frombuffer(v, dtype=np.uint8))
                for v in chosen.fingerprint
            ]
        ).astype(np.float32)
        if not np.array_equal(target, data["fps"]):
            raise ValueError(
                "Training target fingerprints differ from frozen direct candidates"
            )
        paired = paired_rows(train, ids) if multispectrum else ids.copy()
        rows_per_molecule = 2 if multispectrum else 1
        feature_ids = (
            np.column_stack([ids, paired]).reshape(-1) if multispectrum else ids
        )
        chemical = np.stack(
            [
                evidence_features(row, True)
                for row in train.iloc[feature_ids].to_dict("records")
            ]
        )
        distinct_pairs = int((paired != ids).sum()) if multispectrum else 0
        del target, train, chosen
        arrays = {
            name: np.load(FEATURES / (name + ".npy"), mmap_mode="r")
            for name in ["hist", "meta", "loss"]
        }
        encoder.eval().requires_grad_(False).to("cuda")
        weights = torch.load(CRITIC, map_location="cpu", weights_only=True)
        if weights["encoder_sha256"] != digest(ENCODER):
            raise ValueError("Original critic/encoder binding differs")
        base = DirectRanker("fingerprint").eval().to("cuda")
        base.load_state_dict(weights["state_dict"])
        contexts = torch.empty(len(feature_ids), 768, device="cuda")
        queries = torch.empty(len(feature_ids), 128, device="cuda")
        molecules = torch.empty(60000, 128, device="cuda")
        with torch.inference_mode():
            for start in range(0, len(feature_ids), 256):
                if not budget.checkpoint():
                    raise TimeoutError("Query preparation budget exhausted")
                features = torch.cat(
                    [
                        torch.from_numpy(
                            np.array(a[feature_ids[start : start + 256]])
                        ).to("cuda")
                        for a in arrays.values()
                    ],
                    -1,
                )
                latent = encoder.encoder(features)
                contexts[start : start + 256] = latent
                queries[start : start + 256] = base.encode_spectra(latent)
            for start in range(0, 60000, 256):
                molecules[start : start + 256] = base.encode_molecules(
                    torch.from_numpy(data["fps"][start : start + 256]).to("cuda")
                )
        contexts = contexts.reshape(60000, rows_per_molecule, 768)
        queries = queries.reshape(60000, rows_per_molecule, 128)
        chemical = chemical.reshape(60000, rows_per_molecule, DIM)
        chemical = torch.from_numpy(chemical).to("cuda")
        hard = torch.from_numpy(hard).to("cuda")
        model = DiagnosticQueryRanker(base, mask_chemical_channels).to("cuda")
        optimizer = torch.optim.AdamW(
            model.query_residual.parameters(), lr=0.001, weight_decay=0.01
        )
        history = []
        for epoch in range(1, 4):
            model.train()
            ce_losses = []
            anchors = []
            for indexes in np.array_split(rng.permutation(60000), 235):
                if not budget.checkpoint():
                    raise TimeoutError("Query training budget exhausted")
                index = torch.from_numpy(indexes).to("cuda")
                query = (
                    model.query_residual(
                        queries[index].reshape(-1, 128),
                        contexts[index].reshape(-1, 768),
                        chemical[index].reshape(-1, DIM),
                    )
                    .reshape(len(index), rows_per_molecule, 128)
                    .mean(1)
                )
                selected = torch.cat([index[:, None], hard[index]], -1)
                logits = (query[:, None] * molecules[selected]).sum(
                    -1
                ) * model.log_scale.exp().clamp(max=50)
                if strict_mass_negatives:
                    admissible = torch.cat(
                        [
                            torch.ones(
                                (len(index), 1), dtype=torch.bool, device="cuda"
                            ),
                            strict_allowed[index],
                        ],
                        -1,
                    )
                    logits = logits.masked_fill(
                        ~admissible, torch.finfo(logits.dtype).min
                    )
                ce = F.cross_entropy(
                    logits, torch.zeros(len(index), dtype=torch.long, device="cuda")
                )
                anchor = (query - queries[index].mean(1)).square().sum(-1).mean()
                loss = ce + 0.1 * anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite query residual loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.query_residual.parameters(), 1.0)
                optimizer.step()
                ce_losses.append(float(ce.detach()))
                anchors.append(float(anchor.detach()))
            history.append(
                {
                    "epoch": epoch,
                    "listwise_ce": float(np.mean(ce_losses)),
                    "query_anchor": float(np.mean(anchors)),
                }
            )
            print(history[-1], flush=True)
        original = torch.load(CRITIC, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
        if any(
            not torch.equal(v.detach().cpu(), original[k])
            for k, v in base.state_dict().items()
        ):
            raise ValueError("Original critic changed")
        weights["state_dict"] = {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        }
        weights["diagnostic_query_residual"] = {
            "base_critic_sha256": digest(CRITIC),
            "precise": True,
            "dimension": DIM,
            "mask_chemical_channels": mask_chemical_channels,
            "strict_mass_negatives": strict_mass_negatives,
            "multispectrum": multispectrum,
        }
        torch.save(weights, output / "model.pt")
        report = {
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "training_molecules": 60000,
            "train_development_overlap": 0,
            "precise": True,
            "mask_chemical_channels": mask_chemical_channels,
            "strict_mass_negatives": strict_mass_negatives,
            "multispectrum": multispectrum,
            "strict_inside_hard_negatives": strict_inside,
            "distinct_same_ion_training_pairs": distinct_pairs,
            "rows_per_training_molecule": rows_per_molecule,
            "original_critic_tensors_exact": True,
            "original_encoder_frozen": True,
            "history": history,
            "seconds": time.monotonic() - started,
            "model_sha256": digest(output / "model.pt"),
            "scope": "Training diagnostic only; complete paired full-system inference and matched chemical-mask attribution required",
        }
        write_json(output / "report.json", report)
        return report
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-chemical-channels", action="store_true")
    parser.add_argument("--strict-mass-negatives", action="store_true")
    parser.add_argument("--multispectrum", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
