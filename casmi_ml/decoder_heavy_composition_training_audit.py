"""Matched training-only decoder heavy-composition consistency objective."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml.data import write_json
from casmi_ml.diagnostic_bonus_generalization_audit import molecular_bootstrap_gain
from casmi_ml.formula_budget_sampling import atom_count_tokens
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.generation_experiment import conditions, formula_counts, load_model
from casmi_ml.generation_temperature_experiment import CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.training import configure


def molecular_nll(logits, labels):
    mask = labels != 0
    values = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


def heavy_composition_loss(logits, labels, count_tensor, targets):
    mask = (labels != 0)[:, :, None]
    expected = (logits.softmax(-1) @ count_tensor * mask).sum(1)
    denominator = targets.sum(1, keepdim=True)
    if not (denominator > 0).all() or targets[:, 1].any():
        raise ValueError("Positive heavy count targets with hydrogen excluded required")
    return F.smooth_l1_loss(
        expected / denominator, targets / denominator, beta=0.01, reduction="none"
    ).sum(1)


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU training-internal diagnostic required")
    output = Path(output)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError(
            "Deterministic cuBLAS environment required before CUDA initialization"
        )
    torch.use_deterministic_algorithms(True)
    configure(42, threads=1)
    train = pd.read_parquet(
        TRAIN, columns=["inchikey14", "molecular_formula", "normalized_smiles"]
    )
    groups = train.groupby("inchikey14", sort=True)
    keys = list(groups.indices)
    heldout = heldout_mask(keys)
    if len(keys) != 60000 or int(heldout.sum()) != 12006:
        raise ValueError("Original60000 molecular split required")
    cache = conditions(ROOT, "train60k")
    values = np.load(cache / "condition.npy", mmap_mode="r")
    device = torch.device("cuda")
    initial, head, vocabulary = load_model(CHECKPOINT, device)
    xs, labels, sequences, valid = [], [], [], []
    multiple_formulas = []
    for key in keys:
        ids = groups.indices[key]
        group = train.iloc[ids]
        xs.append(np.asarray(values[ids]).mean(0))
        parsed = formula_counts(group.molecular_formula.iloc[0])
        sequence = vocabulary.encode(
            group.normalized_smiles.iloc[0], limit=initial.limit
        )
        labels.append(parsed if parsed is not None else np.zeros(14, np.int64))
        sequences.append(sequence)
        valid.append(parsed is not None and sequence is not None)
        multiple_formulas.append(len(group.molecular_formula.unique()) != 1)
    x = np.asarray(xs, np.float32)
    counts = np.asarray(labels, np.float32) / 32
    valid = np.asarray(valid)
    fit, held = np.flatnonzero(~heldout & valid), np.flatnonzero(heldout & valid)
    count_map, _unsupported = atom_count_tokens(vocabulary)
    count_tensor = torch.tensor(count_map, dtype=torch.float32, device=device)
    heavy_targets = np.asarray(labels, np.float32)
    heavy_targets[:, 1] = 0
    for i in np.flatnonzero(valid):
        token_total = count_map[np.asarray(sequences[i])].sum(0)
        if not np.array_equal(token_total, heavy_targets[i]):
            raise ValueError(
                "All supported label heavy totals must equal training formulas"
            )
    prior_dir = Path(
        "artifacts/research_loop/rounds/0428_decoder_mixed_formula_training_audit"
    )
    prior_protocol = json.loads((prior_dir / "protocol.json").read_text())
    if prior_protocol["source_sha256"] != digest(
        "casmi_ml/decoder_mixed_formula_training_audit.py"
    ) or prior_protocol["train_sha256"] != digest(TRAIN):
        raise ValueError("Source-bound matched428 reference required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "checkpoint_sha256": digest(CHECKPOINT),
            "train_sha256": digest(TRAIN),
            "condition_sha256": digest(cache / "condition.npy"),
            "split_source_sha256": digest(
                "casmi_ml/forward_diagnostic_training_audit.py"
            ),
            "seed": 42,
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.00001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "reference_428_protocol_sha256": digest(prior_dir / "protocol.json"),
            "reference_428_control_model_sha256": digest(
                prior_dir / "private_soft_control_decoder.pt"
            ),
            "auxiliary_weight": 0.1,
            "auxiliary_beta": 0.01,
            "rule": "Original60000/47994fit/12006heldout molecule hash split, first-row canonical SMILES/formula and mean observable conditions, unsupported/unencodable counted with full denominators. Encoder/formula head/initial decoder pretrainedall60K, NOT independent end-to-end generalization. Matched two control replicas/treatment start identical frozen decoder; same deterministic batches/seeds/dropout/update counts/two fixedepochs/no heldout-selected checkpoint. Both arms retain originalsoft conditions, including formulahead. Control pure molecular NLL; treatment adds0.1 normalized heavy-composition SmoothL1 sum with beta.01: expected explicit heavy counts from teacherforced probabilities at nonpadding positions versus training count targets, each divided by recorded total heavy count. Hydrogen excluded. All supported labels verified count-token exact. Prior436 nondeterministic reference replay failure retained. New deterministic algorithms/CUBLAS_WORKSPACE_CONFIG=:4096:8, two independently reseeded softcontrol replicas must reproduce EVERY tensor and every assessed NLL/count-error exactly; no repeateddevelopment checkpoint selection. Formulahead frozen/no gradients. Labels only fit training conditions/targets or final training-side teacherforced diagnostics, no dev/test trueformula generation. Evaluate initial/control/auxiliary on full supported heldout under bothsoft and training-known counts, equal molecule NLL; all12006 denominator and exclusions reported. No activation/structure recall/accuracy claim/extra samples/GAN-DINO expansion. Originalbest236 remains.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    budget = StageBudget(output, "generation", str(output), 7200)
    started = time.monotonic()
    try:
        initial.eval()
        head.eval()
        with torch.inference_mode():
            extras = []
            for offset in range(0, len(x), 256):
                extras.append(
                    head.soft(torch.tensor(x[offset : offset + 256], device=device))
                    .cpu()
                    .numpy()
                )
            soft = np.concatenate(extras)
        original_condition = np.concatenate([x, soft], axis=1)
        true_condition = np.concatenate([x, counts], axis=1)
        initial_state = copy.deepcopy(initial.cpu().state_dict())
        initial.to(device)

        def batch_tokens(ids):
            length = max(len(sequences[i]) for i in ids)
            tokens = torch.zeros((len(ids), length), dtype=torch.long, device=device)
            for slot, i in enumerate(ids):
                tokens[slot, : len(sequences[i])] = torch.tensor(
                    sequences[i], device=device
                )
            return tokens

        def check():
            if c.stopped() or not budget.checkpoint():
                raise RuntimeError(
                    "Mixed formula audit stopped before fixed training/evaluation complete"
                )

        def evaluate(decoder):
            decoder.eval()
            observed = {
                "soft": [],
                "training_integer": [],
                "soft_normalized_heavy_l1": [],
            }
            with torch.inference_mode():
                for offset in range(0, len(held), 32):
                    check()
                    ids = held[offset : offset + 32]
                    tokens = batch_tokens(ids)
                    for name, condition in (
                        ("soft", original_condition),
                        ("training_integer", true_condition),
                    ):
                        logits = decoder(
                            tokens[:, :-1], torch.tensor(condition[ids], device=device)
                        )
                        if name == "soft":
                            targets = torch.tensor(heavy_targets[ids], device=device)
                            expectation = (
                                logits.softmax(-1)
                                @ count_tensor
                                * (tokens[:, 1:] != 0)[:, :, None]
                            ).sum(1)
                            observed["soft_normalized_heavy_l1"].extend(
                                ((expectation - targets).abs().sum(1) / targets.sum(1))
                                .cpu()
                                .tolist()
                            )
                        observed[name].extend(
                            molecular_nll(logits, tokens[:, 1:]).cpu().tolist()
                        )
            return observed

        observations = {"initial": evaluate(initial)}
        history = {}
        control_state = None
        for arm in ("soft_control", "soft_control_repeat", "heavy_auxiliary"):
            configure(42, threads=1)
            initial.load_state_dict(initial_state)
            optimizer = torch.optim.AdamW(
                initial.parameters(), lr=0.00001, weight_decay=0.0001
            )
            rng = np.random.default_rng(42)
            history[arm] = []
            for epoch in range(2):
                order = rng.permutation(fit)
                initial.train()
                total = 0.0
                for offset in range(0, len(order), 32):
                    check()
                    ids = order[offset : offset + 32]
                    tokens = batch_tokens(ids)
                    z = original_condition[ids].copy()
                    logits = initial(tokens[:, :-1], torch.tensor(z, device=device))
                    loss = molecular_nll(logits, tokens[:, 1:]).mean()
                    if arm == "heavy_auxiliary":
                        loss = (
                            loss
                            + 0.1
                            * heavy_composition_loss(
                                logits,
                                tokens[:, 1:],
                                count_tensor,
                                torch.tensor(heavy_targets[ids], device=device),
                            ).mean()
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(initial.parameters(), 1.0)
                    optimizer.step()
                    total += float(loss.detach()) * len(ids)
                history[arm].append(
                    {"epoch": epoch + 1, "fit_molecular_nll": total / len(fit)}
                )
                print(arm, history[arm][-1], flush=True)
            if arm == "soft_control":
                control_state = copy.deepcopy(initial.cpu().state_dict())
                initial.to(device)
            if arm == "soft_control_repeat" and any(
                not torch.equal(v.detach().cpu(), control_state[k])
                for k, v in initial.state_dict().items()
            ):
                raise ValueError(
                    "Deterministic control replicas have different trained tensors"
                )
            torch.save(
                {
                    "decoder": {
                        k: v.detach().cpu() for k, v in initial.state_dict().items()
                    },
                    "initial_checkpoint_sha256": digest(CHECKPOINT),
                },
                output / f"private_{arm}_decoder.pt",
            )
            observations[arm] = evaluate(initial)
            if (
                arm == "soft_control_repeat"
                and observations[arm] != observations["soft_control"]
            ):
                raise ValueError("Deterministic control replicas differ in assessment")
            write_json(output / "private_complete_observations.json", observations)
        report = {
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "fit_molecules": 47994,
            "fit_supported_encodable": len(fit),
            "heldout_molecules": 12006,
            "heldout_supported_encodable": len(held),
            "fit_excluded": int((~heldout & ~valid).sum()),
            "heldout_excluded": int((heldout & ~valid).sum()),
            "all_supported_heavy_label_counts_exact": True,
            "all_deterministic_control_replica_tensors_and_assessments_exact": True,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "multiple_recorded_formula_fit": int(
                (np.asarray(multiple_formulas) & ~heldout).sum()
            ),
            "multiple_recorded_formula_heldout": int(
                (np.asarray(multiple_formulas) & heldout).sum()
            ),
            "arms": {
                name: {k: float(np.mean(v)) for k, v in arm.items()}
                for name, arm in observations.items()
            },
            "auxiliary_over_soft_control_reductions": {
                name: molecular_bootstrap_gain(
                    observations["soft_control"][name],
                    observations["heavy_auxiliary"][name],
                )
                for name in ("soft", "training_integer", "soft_normalized_heavy_l1")
            },
            "history": history,
            "seconds": time.monotonic() - started,
            "scope": "Matched fixed twoepoch finetuning training-only diagnostic on previously all60K-pretrained models. Integer-condition evaluation remains training-only teacher forcing, never oracle dev/test generation. Auxiliary count targets used fit training gradients only. No graph/ranking/independent accuracy claim or deployment; unsupported identities remain in declared denominator.",
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
