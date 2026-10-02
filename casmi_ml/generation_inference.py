"""Unlabeled spectrum-conditioned generation merged into a frozen retrieval CSV."""

import argparse
import hashlib
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from casmi_ml.chemistry import extract_evidence, neutral_mass
from casmi_ml.data import features, write_json
from casmi_ml.generation_experiment import load_model, validate_generated
from casmi_ml.inference import validate_submission
from casmi_ml.metfrag import digest
from casmi_ml.ranking import rrf
from casmi_ml.research_protocol import ENCODER
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


@torch.inference_mode()
def predict(
    checkpoint,
    test_path,
    baseline_csv,
    output,
    routing_csv=None,
    samples=128,
    seconds=1800,
):
    if not 1 <= samples <= 128 or seconds <= 0:
        raise ValueError("Invalid generation resource limits")
    configure(threads=4)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"]["encoder_sha256"] != digest(ENCODER):
        raise ValueError("Generation conditioner checksum mismatch")
    decoder, formula, vocabulary = load_model(checkpoint, device)
    encoder, encoder_checkpoint = load_deployment_checkpoint(ENCODER, "scale")
    encoder.to(device)
    test = pd.read_parquet(test_path)
    base = pd.read_csv(baseline_csv)
    validate_submission(test, base)
    base = base.set_index("molecule_id")
    routing = pd.read_csv(routing_csv or str(baseline_csv) + ".routing.csv")
    if routing.molecule_id.duplicated().any() or set(routing.molecule_id) != set(
        test.molecule_id
    ):
        raise ValueError("Routing IDs do not match test molecules")
    protected = routing.set_index("molecule_id").protected.to_dict()
    if any(not isinstance(v, (bool, np.bool_)) for v in protected.values()):
        raise ValueError("Routing protection must contain booleans")
    rows, audit = [], []
    for molecule_id, group in test.groupby("molecule_id", sort=False):
        smiles = base.loc[molecule_id, "smiles"].split(";")
        status = "protected"
        stats = {}
        if not protected[molecule_id] and time.monotonic() - started < seconds:
            values = [
                features(r, encoder_checkpoint["preprocessing"])
                for r in group.to_dict("records")
            ]
            batch = {
                k: torch.from_numpy(np.stack([v[i] for v in values])).to(device)
                for i, k in enumerate(["hist", "loss", "meta", "peaks", "mask"])
            }
            z = encoder.encoder(
                torch.cat([batch["hist"], batch["meta"], batch["loss"]], -1)
            )
            condition = torch.cat([z, batch["meta"]], -1).mean(0, keepdim=True)
            if saved["config"].get("neutral_mass_condition", False):
                per_spectrum_masses = [
                    neutral_mass(r) for r in group.to_dict("records")
                ]
                mass_features = np.array(
                    [
                        [m / 1250 if m is not None else 0.0, float(m is None)]
                        for m in per_spectrum_masses
                    ],
                    dtype=np.float32,
                )
                condition = torch.cat(
                    [
                        condition,
                        torch.from_numpy(mass_features.mean(0, keepdims=True)).to(
                            device
                        ),
                    ],
                    dim=1,
                )
            hypotheses = formula.hypotheses(condition)
            # Seed each molecule independently of group processing/fallback order.
            seed = int.from_bytes(
                hashlib.sha256(str(molecule_id).encode()).digest()[:8], "big"
            ) % (2**63 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)
            sequence, logp, finished = decoder.generate(
                torch.cat([condition, formula.soft(condition)], 1),
                samples,
                generator=generator,
            )
            masses = [
                m
                for r in group.to_dict("records")
                if (m := neutral_mass(r)) is not None
            ]
            mass = float(np.median(masses)) if masses else None
            evidence = [extract_evidence(r) for r in group.to_dict("records")]
            candidates, stats = validate_generated(
                sequence.cpu().tolist(),
                logp.cpu().tolist(),
                finished.cpu().tolist(),
                vocabulary,
                mass,
                hypotheses,
                evidence,
            )
            if candidates:
                lookup = {
                    Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]: s for s in smiles
                }
                original = list(lookup)
                generated = [c["key"] for c in candidates]
                lookup.update(
                    {
                        c["key"]: c["smiles"]
                        for c in candidates
                        if c["key"] not in lookup
                    }
                )
                rank = rrf([original, generated], [0.75, 0.25])
                smiles = [lookup[k] for k in rank[:25]]
                status = "generated_and_merged"
            else:
                status = "no_valid_mass_matching_generation"
        elif not protected[molecule_id]:
            status = "budget_retrieval_fallback"
        rows.append({"molecule_id": molecule_id, "smiles": ";".join(smiles)})
        audit.append({"molecule_id": molecule_id, "status": status, **stats})
    submission = pd.DataFrame(rows)
    validate_submission(test, submission)
    submission.to_csv(output, index=False)
    pd.DataFrame(audit).to_csv(str(output) + ".generation.csv", index=False)
    write_json(
        str(output) + ".report.json",
        {
            "molecules": len(rows),
            "seconds": time.monotonic() - started,
            "checkpoint_sha256": digest(checkpoint),
            "baseline_sha256": digest(baseline_csv),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "formula_oracle_used": False,
            "kaggle_submitted": False,
        },
    )
    return submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test", type=Path, default=Path("data/test.parquet"))
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--routing", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seconds", type=float, default=1800)
    a = p.parse_args()
    predict(a.checkpoint, a.test, a.baseline, a.output, a.routing, a.samples, a.seconds)


if __name__ == "__main__":
    main()
