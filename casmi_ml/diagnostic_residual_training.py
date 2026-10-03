"""Matched coarse/exact diagnostic evidence residuals, original60K labels only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import DIM, DiagnosticResidual, evidence_features
from casmi_ml.encoder_pair_finetuning import FEATURES
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def run(output, precise=False, contextual=False, mask_chemical_channels=False):
    if mask_chemical_channels and (not contextual or not precise):
        raise ValueError(
            "Matched chemical-channel mask requires precise contextual arm"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure(42, threads=4)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "feature_source_sha256": digest("casmi_ml/diagnostic_residual.py"),
            "chemistry_source_sha256": digest("casmi_ml/chemistry.py"),
            "encoder_sha256": digest(ENCODER),
            "training_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "precise": precise,
            "contextual": contextual,
            "mask_chemical_channels": mask_chemical_channels,
            "width": 64,
            "feature_dim": DIM,
            "epochs": 3,
            "lr": 0.001,
            "base_encoder_frozen": True,
            "row_selection": "one rng42 representative per original60K training molecule, identical arms",
            "training_objective": "fingerprint BCE on frozen base logits plus zero initialized evidence residual",
            "same_control": "same network, labels, selected spectra, optimizer and epochs; absolute tolerance .5 vs .002 Da",
            "training_seconds": 86400,
            "holdout_used": False,
            "independent_acceptance": False,
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    budget = StageBudget(
        output,
        "representation",
        "diagnostic_context_residual_fixed3"
        if contextual
        else "diagnostic_residual_fixed3",
        86400,
        lock_path=output / "representation.lock",
    )
    started = time.monotonic()
    try:
        train = pd.read_parquet(TRAIN)
        dev = set(
            pd.read_parquet(
                ROOT / "researchdev.parquet", columns=["inchikey14"]
            ).inchikey14
        )
        groups = train.groupby("inchikey14", sort=True).indices
        if len(groups) != 60000 or set(groups) & dev:
            raise ValueError("Original60K isolation failed")
        info = pd.read_parquet(FEATURES / "rows.parquet")
        if not info.equals(train[["inchikey14", "row_id"]].reset_index(drop=True)):
            raise ValueError("Feature row mapping differs")
        rng = np.random.default_rng(42)
        ids = np.array([rng.choice(v) for v in groups.values()])
        chosen = train.iloc[ids]
        chemical = np.zeros((60000, DIM), np.float32)
        for i, row in enumerate(chosen.to_dict("records")):
            if not budget.checkpoint():
                raise TimeoutError("Preparation budget exhausted")
            chemical[i] = evidence_features(row, precise)
            if (i + 1) % 5000 == 0:
                print(
                    "evidence", i + 1, "seconds", time.monotonic() - started, flush=True
                )
        arrays = {
            name: np.load(FEATURES / (name + ".npy"), mmap_mode="r")
            for name in ["hist", "meta", "loss"]
        }
        base, saved = load_deployment_checkpoint(ENCODER, "scale")
        base.eval().to("cuda")
        baseline = np.zeros((60000, 2048), np.float32)
        contexts = np.zeros((60000, 768), np.float32) if contextual else None
        with torch.inference_mode():
            for start in range(0, len(ids), 256):
                batch = {
                    name: torch.from_numpy(np.array(a[ids[start : start + 256]])).to(
                        "cuda"
                    )
                    for name, a in arrays.items()
                }
                if contextual:
                    latent = base.encoder(
                        torch.cat([batch["hist"], batch["meta"], batch["loss"]], -1)
                    )
                    contexts[start : start + 256] = latent.cpu().numpy()
                    baseline[start : start + 256] = base.head(latent).cpu().numpy()
                else:
                    baseline[start : start + 256] = base(batch).cpu().numpy()
        targets = np.stack(
            [
                np.unpackbits(np.frombuffer(v, dtype=np.uint8))
                for v in chosen.fingerprint
            ]
        ).astype(np.float32)
        model = DiagnosticResidual(
            base, contextual=contextual, mask_chemical_channels=mask_chemical_channels
        ).to("cuda")
        optimizer = torch.optim.AdamW(
            model.correction.parameters(), lr=0.001, weight_decay=0.01
        )
        history = []
        for epoch in range(1, 4):
            model.train()
            losses = []
            for indexes in np.array_split(rng.permutation(60000), 235):
                if not budget.checkpoint():
                    raise TimeoutError("Training budget exhausted")
                x = torch.from_numpy(chemical[indexes]).to("cuda")
                y = torch.from_numpy(targets[indexes]).to("cuda")
                logits = torch.from_numpy(baseline[indexes]).to(
                    "cuda"
                ) + model.residual(
                    x,
                    torch.from_numpy(contexts[indexes]).to("cuda")
                    if contextual
                    else None,
                )
                loss = F.binary_cross_entropy_with_logits(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append({"epoch": epoch, "bce": float(np.mean(losses))})
            print(history[-1], flush=True)
        base_original, _ = load_deployment_checkpoint(ENCODER, "scale")
        if any(
            not torch.equal(value.detach().cpu(), base_original.state_dict()[key])
            for key, value in base.state_dict().items()
        ):
            raise ValueError("Frozen original encoder changed")
        saved["state_dict"] = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        saved["diagnostic_residual"] = {
            "precise": precise,
            "dimension": DIM,
            "contextual": contextual,
            "mask_chemical_channels": mask_chemical_channels,
            "base_sha256": digest(ENCODER),
        }
        saved["preprocessing"] = dict(
            saved["preprocessing"], diagnostic_precision=precise
        )
        torch.save(saved, output / "model.pt")
        report = {
            "diagnostic_only": True,
            "training_molecules": 60000,
            "selected_spectra": 60000,
            "train_development_overlap": 0,
            "precise": precise,
            "base_encoder_exact": True,
            "contextual": contextual,
            "mask_chemical_channels": mask_chemical_channels,
            "evidence_nonempty": int((chemical.max(1) > 0).sum()),
            "seconds": time.monotonic() - started,
            "encoder_sha256": digest(output / "model.pt"),
            "history": history,
            "independent_acceptance": False,
            "scope": "Training diagnostics only; complete proposal and final181 system evaluation still required",
        }
        write_json(output / "report.json", report)
        return report
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precise", action="store_true")
    p.add_argument("--contextual", action="store_true")
    p.add_argument("--mask-chemical-channels", action="store_true")
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
