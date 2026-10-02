"""Unlabeled spectrum-conditioned generation merged into a frozen retrieval CSV."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from casmi_ml.chemistry import extract_evidence, neutral_mass
from casmi_ml.data import write_json
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
    encoder_path=None,
    prefix=None,
    slots=None,
    full_rankings=None,
):
    encoder_path = Path(encoder_path or ENCODER)
    if not 1 <= samples <= 128 or seconds <= 0:
        raise ValueError("Invalid generation resource limits")
    if prefix is not None and (
        slots is None or prefix < 1 or slots < 1 or full_rankings is None
    ):
        raise ValueError(
            "Explicit slots require positive values and full retrieval rankings"
        )
    configure(threads=4)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"]["encoder_sha256"] != digest(encoder_path):
        raise ValueError("Generation conditioner checksum mismatch")
    decoder, formula, vocabulary = load_model(checkpoint, device)
    encoder, encoder_checkpoint = load_deployment_checkpoint(encoder_path, "scale")
    encoder.to(device)
    test = pd.read_parquet(test_path)
    base = pd.read_csv(baseline_csv)
    validate_submission(test, base)
    base = base.set_index("molecule_id")
    full = None
    if full_rankings is not None:
        raw = json.loads(Path(full_rankings).read_text())
        full = {r["molecule_id"]: r["smiles"] for r in raw}
        if len(full) != len(raw) or set(full) != set(test.molecule_id):
            raise ValueError("Full retrieval ranking IDs differ")
        for molecule_id, values in full.items():
            if values[:25] != base.loc[molecule_id, "smiles"].split(";"):
                raise ValueError("Full rankings do not match retrieval submission")
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
        smiles = (
            full[molecule_id]
            if full is not None
            else base.loc[molecule_id, "smiles"].split(";")
        )[:]
        original_top25 = smiles[:25]
        status = "protected"
        stats = {}
        if not protected[molecule_id] and time.monotonic() - started < seconds:
            from casmi_ml.generation_sampling import condition_for_group

            condition = condition_for_group(
                encoder,
                encoder_checkpoint["preprocessing"],
                group,
                device,
                saved["config"].get("neutral_mass_condition", False),
            )
            hypotheses = formula.hypotheses(condition)
            # Seed each molecule independently of group processing/fallback order.
            from casmi_ml.generation_sampling import sampling_seed

            seed = sampling_seed(group)
            generator = torch.Generator(device=device).manual_seed(seed)
            try:
                sequence, logp, finished = decoder.generate(
                    torch.cat([condition, formula.soft(condition)], 1),
                    samples,
                    generator=generator,
                    deadline=started + seconds,
                )
            except TimeoutError:
                rows.append(
                    {"molecule_id": molecule_id, "smiles": ";".join(original_top25)}
                )
                audit.append(
                    {"molecule_id": molecule_id, "status": "budget_retrieval_fallback"}
                )
                continue
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
                if prefix is None:
                    rank = rrf([original, generated], [0.75, 0.25])
                else:
                    from casmi_ml.generation_slots import insert_generated

                    rank = insert_generated(original, generated, prefix, slots)
                smiles = [lookup[k] for k in rank[:25]]
                status = "generated_and_merged"
            else:
                status = "no_valid_mass_matching_generation"
        elif not protected[molecule_id]:
            status = "budget_retrieval_fallback"
        smiles = smiles[:25] if status == "generated_and_merged" else original_top25
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
            "sampling": "label_independent_spectrum_hash_v1",
            "prefix": prefix,
            "slots": slots,
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
    p.add_argument("--encoder", type=Path, default=ENCODER)
    p.add_argument("--prefix", type=int)
    p.add_argument("--slots", type=int)
    p.add_argument("--full-rankings", type=Path)
    a = p.parse_args()
    if (a.prefix is None) != (a.slots is None):
        p.error("--prefix and --slots must be provided together")
    predict(
        a.checkpoint,
        a.test,
        a.baseline,
        a.output,
        a.routing,
        a.samples,
        a.seconds,
        a.encoder,
        a.prefix,
        a.slots,
        a.full_rankings,
    )


if __name__ == "__main__":
    main()
