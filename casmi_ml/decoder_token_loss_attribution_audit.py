"""Training-only token attribution after rejected matched decoder graph pilot."""

import argparse
import copy
import json
import time
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

TRAINED = Path(
    "artifacts/research_loop/rounds/0428_decoder_mixed_formula_training_audit"
)
CATEGORIES = ("atom", "bond_separator", "branch", "ring", "stereo", "eos")


def token_category(token):
    if token == "<eos>":
        return "eos"
    if token in ("(", ")"):
        return "branch"
    if token.isdigit() or (token.startswith("%") and token[1:].isdigit()):
        return "ring"
    if token in ("@", "@@"):
        return "stereo"
    if token in ("-", "=", "#", ":", "/", "\\", "."):
        return "bond_separator"
    if token.startswith("[") or token.isalpha():
        return "atom"
    raise ValueError("Unclassified nonpadding SMILES token: " + token)


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU attribution required")
    output = Path(output)
    prior = json.loads((TRAINED / "protocol.json").read_text())
    prior_observed = json.loads(
        (TRAINED / "private_complete_observations.json").read_text()
    )
    if (
        prior["source_sha256"]
        != digest("casmi_ml/decoder_mixed_formula_training_audit.py")
        or prior["train_sha256"] != digest(TRAIN)
        or prior["checkpoint_sha256"] != digest(CHECKPOINT)
    ):
        raise ValueError("Original source-bound matched training required")
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
            "training_protocol_sha256": digest(TRAINED / "protocol.json"),
            "training_observations_sha256": digest(
                TRAINED / "private_complete_observations.json"
            ),
            "checkpoint_sha256": digest(CHECKPOINT),
            "train_sha256": digest(TRAIN),
            "condition_sha256": digest(cache / "condition.npy"),
            "models": {
                name: digest(TRAINED / ("private_" + name + "_decoder.pt"))
                for name in ("soft_control", "mixed_integer")
            },
            "split_source_sha256": digest(
                "casmi_ml/forward_diagnostic_training_audit.py"
            ),
            "categories": CATEGORIES,
            "rule": "No training, generation, new model selection or dev/test inputs. All original12006 training-side heldout keys; same supported12002, canonical first-row target, mean observed encoder plus originalsoft formula, batch32, seed42, original/frozen matchedsoft/mixed models. Initial encoder/head/decoder pretrainedall60K, not independent generalization. Require all three originalsoft molecular NLL arrays exactly reproduce428 before acceptance. Every nonpadding label exactly one category, token sums reconstruct full loss. Category NLL is equal eligible-molecule mean, denominators explicit; paired molecular bootstrap. Bracket tokens counted atom even when stereo annotated; no claim that this isolates all stereochemistry. No deployment or structure accuracy claim;429 rejection retained.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    budget = StageBudget(output, "generation", str(output), 1800)
    started = time.monotonic()
    try:
        device = torch.device("cuda")
        decoder, head, vocabulary = load_model(CHECKPOINT, device)
        original_state = copy.deepcopy(decoder.cpu().state_dict())
        decoder.to(device)
        selected, xs, sequences = [], [], []
        for i in np.flatnonzero(heldout):
            group = train.iloc[groups.indices[keys[i]]]
            sequence = vocabulary.encode(
                group.normalized_smiles.iloc[0], limit=decoder.limit
            )
            if (
                formula_counts(group.molecular_formula.iloc[0]) is not None
                and sequence is not None
            ):
                selected.append(int(i))
                xs.append(np.asarray(values[groups.indices[keys[i]]]).mean(0))
                sequences.append(sequence)
        if len(selected) != 12002:
            raise ValueError("Original supported denominator changed")
        mapping = torch.tensor(
            [
                -1
                if t in ("<pad>", "<bos>", "<unk>")
                else CATEGORIES.index(token_category(t))
                for t in vocabulary.tokens
            ],
            device=device,
        )
        x = np.asarray(xs, np.float32)
        with torch.inference_mode():
            extras = []
            # Preserve428 original all60000 batch boundaries for exact formulahead values.
            all_x = np.asarray(
                [np.asarray(values[groups.indices[key]]).mean(0) for key in keys],
                np.float32,
            )
            for offset in range(0, len(all_x), 256):
                extras.append(
                    head.soft(torch.tensor(all_x[offset : offset + 256], device=device))
                    .cpu()
                    .numpy()
                )
            soft = np.concatenate(extras)[selected]
        cond = np.concatenate([x, soft], 1)
        observed = {}
        counts = None
        for name in ("initial", "soft_control", "mixed_integer"):
            decoder.load_state_dict(
                original_state
                if name == "initial"
                else torch.load(
                    TRAINED / ("private_" + name + "_decoder.pt"),
                    map_location="cpu",
                    weights_only=True,
                )["decoder"]
            )
            decoder.eval()
            arm = {
                "full_nll": [],
                "category_loss_sums": [],
                "category_token_counts": [],
            }
            with torch.inference_mode():
                for offset in range(0, len(selected), 32):
                    if c.stopped() or not budget.checkpoint():
                        raise RuntimeError(
                            "Attribution stopped before complete assessment"
                        )
                    batch = sequences[offset : offset + 32]
                    tokens = torch.zeros(
                        (len(batch), max(map(len, batch))),
                        dtype=torch.long,
                        device=device,
                    )
                    for j, sequence in enumerate(batch):
                        tokens[j, : len(sequence)] = torch.tensor(
                            sequence, device=device
                        )
                    labels = tokens[:, 1:]
                    logits = decoder(
                        tokens[:, :-1],
                        torch.tensor(cond[offset : offset + 32], device=device),
                    )
                    loss = F.cross_entropy(
                        logits.transpose(1, 2), labels, reduction="none"
                    )
                    mask = labels != 0
                    category = mapping[labels]
                    if not ((category >= 0) | ~mask).all():
                        raise ValueError("Nonpadding label not classified")
                    token_counts = torch.stack(
                        [(category == k).sum(1) for k in range(len(CATEGORIES))], 1
                    )
                    sums = torch.stack(
                        [
                            (loss * (category == k)).sum(1)
                            for k in range(len(CATEGORIES))
                        ],
                        1,
                    )
                    if not torch.equal(
                        token_counts.sum(1), mask.sum(1)
                    ) or not torch.allclose(
                        sums.sum(1), (loss * mask).sum(1), atol=1e-5, rtol=1e-6
                    ):
                        raise ValueError(
                            "Category partition does not reconstruct full loss"
                        )
                    arm["full_nll"].extend(
                        ((loss * mask).sum(1) / mask.sum(1)).cpu().tolist()
                    )
                    arm["category_loss_sums"].extend(sums.cpu().tolist())
                    arm["category_token_counts"].extend(token_counts.cpu().tolist())
            if arm["full_nll"] != prior_observed[name]["soft"]:
                write_json(output / "private_failed_observations.json", arm)
                raise ValueError(
                    "Molecular NLL differs from frozen428 originalsoft assessment"
                )
            this_counts = np.asarray(arm["category_token_counts"])
            if counts is not None and not np.array_equal(counts, this_counts):
                raise ValueError("Arms changed token denominators")
            counts = this_counts
            observed[name] = arm
            print("Complete exact molecular NLL arm", name, flush=True)
        write_json(output / "private_complete_observations.json", observed)
        metrics, comparisons = {}, {}
        for name, arm in observed.items():
            sums = np.asarray(arm["category_loss_sums"])
            metrics[name] = {
                "full_nll": float(np.mean(arm["full_nll"])),
                "categories": {
                    category: {
                        "molecules_with_tokens": int((counts[:, k] > 0).sum()),
                        "tokens": int(counts[:, k].sum()),
                        "eligible_molecular_nll": float(
                            np.mean(
                                sums[counts[:, k] > 0, k] / counts[counts[:, k] > 0, k]
                            )
                        )
                        if (counts[:, k] > 0).any()
                        else None,
                    }
                    for k, category in enumerate(CATEGORIES)
                },
            }
        for base, treatment in (
            ("initial", "soft_control"),
            ("initial", "mixed_integer"),
            ("soft_control", "mixed_integer"),
        ):
            a = np.asarray(observed[base]["category_loss_sums"])
            b = np.asarray(observed[treatment]["category_loss_sums"])
            comparisons[treatment + "_over_" + base] = {
                category: molecular_bootstrap_gain(
                    a[counts[:, k] > 0, k] / counts[counts[:, k] > 0, k],
                    b[counts[:, k] > 0, k] / counts[counts[:, k] > 0, k],
                )
                if (counts[:, k] > 0).any()
                else None
                for k, category in enumerate(CATEGORIES)
            }
        report = {
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "heldout_molecules": 12006,
            "supported_encodable": len(selected),
            "excluded": 4,
            "all_three_originalsoft_molecular_nll_arrays_exact428": True,
            "all_nonpadding_labels_partitioned": True,
            "arms": metrics,
            "paired_category_nll_reductions": comparisons,
            "seconds": time.monotonic() - started,
            "scope": "Training-side teacherforced token attribution, all models initially pretrainedall60K. Category eligible molecular denominators explicit; bracket stereo belongs atom category. No free generation, structural recall, model promotion or independent acceptance; rejected429 unchanged.",
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
