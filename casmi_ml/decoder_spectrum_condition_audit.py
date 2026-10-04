"""Frozen training-side decoder spectrum dependence with formula fixed."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import write_json
from casmi_ml.decoder_formula_response_audit import token_metrics
from casmi_ml.decoder_token_loss_attribution_audit import TRAINED
from casmi_ml.diagnostic_bonus_generalization_audit import molecular_bootstrap_gain
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.generation_experiment import conditions, formula_counts, load_model
from casmi_ml.generation_temperature_experiment import CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.training import configure


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU frozen spectrum diagnostic required")
    output = Path(output)
    configure(42, threads=1)
    train = pd.read_parquet(
        TRAIN, columns=["inchikey14", "molecular_formula", "normalized_smiles"]
    )
    groups = train.groupby("inchikey14", sort=True)
    keys = list(groups.indices)
    heldout = heldout_mask(keys)
    if len(keys) != 60000 or int(heldout.sum()) != 12006:
        raise ValueError("Original fixed molecular split required")
    cache = conditions(ROOT, "train60k")
    values = np.load(cache / "condition.npy", mmap_mode="r")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "checkpoint_sha256": digest(CHECKPOINT),
            "train_sha256": digest(TRAIN),
            "condition_sha256": digest(cache / "condition.npy"),
            "encoder_sha256": digest(ENCODER),
            "reference_observations_sha256": digest(
                TRAINED / "private_complete_observations.json"
            ),
            "split_source_sha256": digest(
                "casmi_ml/forward_diagnostic_training_audit.py"
            ),
            "metrics_source_sha256": digest(
                "casmi_ml/decoder_formula_response_audit.py"
            ),
            "rule": "All12006 fixed training-side heldout representatives, original supported12002, canonical teacherforced labels. Original encoder/head/decoder pretrainedall60K, not independent end-to-end generalization. No fitting/dev/test/generation. Keep query originalsoft formula AND all103 metadata and2 mass channels fixed in all interventions. Change ONLY first768 encoder latent channels. Original latent z, zero latent z, next full sorted heldout-supported molecule z (cyclic shuffle), and lexicographically first fit-side DIFFERENT identity with exact recorded count-vector formula. Formula labels only select training-side diagnostic donor; never inference inputs. Missing sameformula donors use original z internally but excluded explicitly from donor metrics; all12006 denominator retained. Mean original observable condition per molecule, latent768/meta103/mass2 verified from checkpoints, batch32, seed42. Require full original molecular NLL exactly428 before reporting. NLL/KL/argmax response descriptive; no deployment, architecture or graph-recall acceptance.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    budget = StageBudget(output, "generation", str(output), 1800)
    started = time.monotonic()
    try:
        device = torch.device("cuda")
        decoder, head, vocabulary = load_model(CHECKPOINT, device)
        xs, formulas, sequences = [], [], []
        for key in keys:
            ids = groups.indices[key]
            group = train.iloc[ids]
            xs.append(np.asarray(values[ids]).mean(0))
            parsed = formula_counts(group.molecular_formula.iloc[0])
            formulas.append(None if parsed is None else tuple(int(v) for v in parsed))
            sequences.append(
                vocabulary.encode(group.normalized_smiles.iloc[0], limit=decoder.limit)
            )
        x = np.asarray(xs, np.float32)
        selected = np.asarray(
            [
                i
                for i in np.flatnonzero(heldout)
                if formulas[i] is not None and sequences[i] is not None
            ]
        )
        if len(selected) != 12002:
            raise ValueError("Original supported denominator changed")
        donors = {}
        for i in np.flatnonzero(~heldout):
            if formulas[i] is not None:
                donors.setdefault(formulas[i], int(i))
        donor = np.asarray([donors.get(formulas[i], int(i)) for i in selected])
        eligible = donor != selected
        if any(heldout[i] for i in donor[eligible]) or any(
            formulas[i] != formulas[j]
            for i, j in zip(selected[eligible], donor[eligible])
        ):
            raise ValueError(
                "Diagnostic donors must be fit-only distinct exact formula identities"
            )
        with torch.inference_mode():
            extra = []
            for offset in range(0, len(x), 256):
                extra.append(
                    head.soft(torch.tensor(x[offset : offset + 256], device=device))
                    .cpu()
                    .numpy()
                )
        soft = np.concatenate(extra)[selected]
        encoder_info = torch.load(ENCODER, map_location="cpu", weights_only=True)
        if (
            encoder_info["metadata_dim"] != 103
            or x.shape[1] != 873
            or head.net[0].in_features != 873
        ):
            raise ValueError("Original latent768/meta103/mass2 layout required")
        conditions_by_name = {}
        for name, latent in [
            ("original", x[selected, :768]),
            ("zero_spectrum", np.zeros_like(x[selected, :768])),
            ("shuffled_spectrum", np.roll(x[selected, :768], -1, axis=0)),
            ("same_formula_fit_spectrum", x[donor, :768]),
        ]:
            z = x[selected].copy()
            z[:, :768] = latent
            conditions_by_name[name] = np.concatenate([z, soft], 1)
            if not np.array_equal(
                conditions_by_name[name][:, 768:],
                conditions_by_name["original"][:, 768:],
            ):
                raise ValueError(
                    "Intervention changed query metadata/mass/formula channels"
                )
        observed = {
            name: {
                metric: [] for metric in ("nll", "base_to_arm_kl", "argmax_change_rate")
            }
            for name in conditions_by_name
        }
        with torch.inference_mode():
            for offset in range(0, len(selected), 32):
                if c.stopped() or not budget.checkpoint():
                    raise RuntimeError(
                        "Spectrum diagnostic stopped before complete cohort"
                    )
                batch = [sequences[i] for i in selected[offset : offset + 32]]
                tokens = torch.zeros(
                    (len(batch), max(map(len, batch))), dtype=torch.long, device=device
                )
                for j, seq in enumerate(batch):
                    tokens[j, : len(seq)] = torch.tensor(seq, device=device)
                reference = decoder(
                    tokens[:, :-1],
                    torch.tensor(
                        conditions_by_name["original"][offset : offset + 32],
                        device=device,
                    ),
                )
                for name, z in conditions_by_name.items():
                    logits = (
                        reference
                        if name == "original"
                        else decoder(
                            tokens[:, :-1],
                            torch.tensor(z[offset : offset + 32], device=device),
                        )
                    )
                    metrics = token_metrics(logits, reference, tokens[:, 1:])
                    for metric, v in metrics.items():
                        observed[name][metric].extend(v.cpu().tolist())
        prior = json.loads((TRAINED / "private_complete_observations.json").read_text())
        if observed["original"]["nll"] != prior["initial"]["soft"]:
            write_json(output / "private_failed_observations.json", observed)
            raise ValueError("Original full molecular NLL differs from frozen428")
        write_json(
            output / "private_complete_observations.json",
            {
                "molecular_metrics": observed,
                "same_formula_donor_eligible": eligible.tolist(),
            },
        )
        arms = {}
        comparison = {}
        for name, metrics in observed.items():
            mask = (
                eligible
                if name == "same_formula_fit_spectrum"
                else np.ones(len(selected), bool)
            )
            arms[name] = {
                "molecules": int(mask.sum()),
                **{m: float(np.asarray(v)[mask].mean()) for m, v in metrics.items()},
            }
            if name != "original":
                comparison[name] = molecular_bootstrap_gain(
                    np.asarray(observed["original"]["nll"])[mask],
                    np.asarray(metrics["nll"])[mask],
                )
        arms["original_same_formula_eligible"] = {
            "molecules": int(eligible.sum()),
            "nll": float(np.asarray(observed["original"]["nll"])[eligible].mean()),
        }
        report = {
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "heldout_molecules": 12006,
            "supported_encodable": 12002,
            "excluded": 4,
            "same_formula_fit_donor_available": int(eligible.sum()),
            "same_formula_fit_donor_missing": int((~eligible).sum()),
            "original_molecular_nll_exact428": True,
            "query_metadata_mass_softformula_channels_exact_allarms": True,
            "arms": arms,
            "paired_nll_reductions": comparison,
            "seconds": time.monotonic() - started,
            "scope": "Frozen training-side spectrum intervention with query originalsoft formula,103 metadata and2 mass channels fixed, only768 latent channels changed. Formula labels used ONLY to choose fit-side exact-formula diagnostic donors; no dev/test/oracle generation or model selection. All pretrainedall60K; cannot establish independent structural recall or architecture gain.",
        }
        write_json(output / "report.json", report)
        print(json.dumps(report), flush=True)
        return report
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
