"""Training-only teacher-forced decoder response to formula condition replacement."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml.data import write_json
from casmi_ml.diagnostic_bonus_generalization_audit import molecular_bootstrap_gain
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.generation_experiment import conditions, formula_counts, load_model
from casmi_ml.generation_temperature_experiment import CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.training import configure


def token_metrics(logits, reference, labels):
    mask = labels != 0
    loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    logp, base = logits.log_softmax(-1), reference.log_softmax(-1)
    kl = (base.exp() * (base - logp)).sum(-1)
    changed = logits.argmax(-1) != reference.argmax(-1)
    denominator = mask.sum(1).clamp_min(1)
    return {
        "nll": (loss * mask).sum(1) / denominator,
        "base_to_arm_kl": (kl * mask).sum(1) / denominator,
        "argmax_change_rate": (changed * mask).sum(1) / denominator,
    }


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only training decoder response diagnostic required")
    output = Path(output)
    configure(42, threads=1)
    train = pd.read_parquet(
        TRAIN, columns=["inchikey14", "molecular_formula", "normalized_smiles"]
    )
    groups = train.groupby("inchikey14", sort=True)
    keys = list(groups.indices)
    ids = np.flatnonzero(heldout_mask(keys))[:200]
    cache = conditions(ROOT, "train60k")
    values = np.load(cache / "condition.npy", mmap_mode="r")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "checkpoint_sha256": digest(CHECKPOINT),
            "training_sha256": digest(TRAIN),
            "condition_sha256": digest(cache / "condition.npy"),
            "split_source_sha256": digest(
                "casmi_ml/forward_diagnostic_training_audit.py"
            ),
            "rule": "Fixed first200 original training-internal heldout-side molecular keys, original mean observable condition and canonical recorded SMILES teacher forcing. Frozen decoder/encoder/formula head already trained all60K, NOT independent generalization. Prespecified conditions: originalsoft counts, training-known truecounts/32 (diagnostic ONLY), zero formula, next fixed cohort identity truecounts/32 shuffled. Labels only training targets and diagnostic teacher-forcing conditions; no dev/test/oracle generation or new deployment. No parameter updates, settings selection or expansion. All200 denominator and count unsupported/token-unencodable separately; compare NLL/KL/token-argmax response and memory feature RMS, not structural recall or inference accuracy.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    budget = StageBudget(output, "generation", str(output), 1800)
    try:
        device = torch.device("cuda")
        decoder, head, vocabulary = load_model(CHECKPOINT, device)
        zs, truths, sequences, valid = [], [], [], []
        for i in ids:
            group = train.iloc[groups.indices[keys[i]]]
            zs.append(np.asarray(values[groups.indices[keys[i]]]).mean(0))
            formula = formula_counts(group.molecular_formula.iloc[0])
            sequence = vocabulary.encode(
                group.normalized_smiles.iloc[0], limit=decoder.limit
            )
            truths.append(formula if formula is not None else np.zeros(14, np.int64))
            sequences.append(sequence)
            valid.append(formula is not None and sequence is not None)
        z = torch.tensor(np.asarray(zs), device=device)
        truth = (
            torch.tensor(np.asarray(truths), dtype=torch.float32, device=device) / 32
        )
        shuffled = truth.roll(-1, dims=0)
        observed = {
            name: {
                metric: []
                for metric in (
                    "nll",
                    "base_to_arm_kl",
                    "argmax_change_rate",
                    "memory_delta_rms",
                )
            }
            for name in (
                "original",
                "training_true_formula",
                "zero_formula",
                "shuffled_training_formula",
            )
        }
        with torch.inference_mode():
            soft = head.soft(z)
            conditions_by_name = {
                name: torch.cat([z, extra], 1)
                for name, extra in (
                    ("original", soft),
                    ("training_true_formula", truth),
                    ("zero_formula", torch.zeros_like(soft)),
                    ("shuffled_training_formula", shuffled),
                )
            }
            selected = np.flatnonzero(valid)
            for start in range(0, len(selected), 8):
                if c.stopped() or not budget.checkpoint():
                    raise RuntimeError("Response audit stopped before fixed200")
                batch = selected[start : start + 8]
                length = max(len(sequences[i]) for i in batch)
                tokens = torch.zeros(
                    (len(batch), length), dtype=torch.long, device=device
                )
                for slot, i in enumerate(batch):
                    tokens[slot, : len(sequences[i])] = torch.tensor(
                        sequences[i], device=device
                    )
                base = decoder(tokens[:, :-1], conditions_by_name["original"][batch])
                memory = decoder.condition(conditions_by_name["original"][batch])
                for name, condition in conditions_by_name.items():
                    logits = (
                        base
                        if name == "original"
                        else decoder(tokens[:, :-1], condition[batch])
                    )
                    metrics = token_metrics(logits, base, tokens[:, 1:])
                    metrics["memory_delta_rms"] = (
                        (decoder.condition(condition[batch]) - memory)
                        .square()
                        .mean(1)
                        .sqrt()
                    )
                    for metric, x in metrics.items():
                        observed[name][metric].extend(x.cpu().tolist())
        write_json(output / "private_molecular_observations.json", observed)
        arms = {
            name: {metric: float(np.mean(x)) for metric, x in metrics.items()}
            for name, metrics in observed.items()
        }
        comparisons = {
            name: molecular_bootstrap_gain(observed["original"]["nll"], metrics["nll"])
            for name, metrics in observed.items()
            if name != "original"
        }
        result = {
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "molecules": 200,
            "teacher_forced_supported_molecules": int(sum(valid)),
            "unsupported_or_unencodable_molecules": int(200 - sum(valid)),
            "arms": arms,
            "paired_nll_reductions": comparisons,
            "scope": "Frozen training-only decoder sensitivity observation. Training trueformula is only a diagnostic intervention, not inference proposal/structure generation. All models pretrainedall60K; no independent generalization or scientific promotion. NLL/KL response cannot establish graph recall gain.",
        }
        write_json(output / "report.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
