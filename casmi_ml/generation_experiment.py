"""Mass-conditioned SMILES generation with predicted, never oracle, formula conditions."""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from casmi_ml.chemistry import extract_evidence, neutral_mass, score_candidate
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_models import SmilesDecoder, SmilesVocabulary
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze, prepare
from casmi_ml.scale_experiment import SelectedFeatures
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

ELEMENTS = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si", "B", "Na", "K"]
MAX_COUNT = 256


def formula_counts(formula):
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if "".join(e + n for e, n in parts) != formula or any(
        e not in ELEMENTS for e, n in parts
    ):
        return None
    counts = {e: int(n or 1) for e, n in parts}
    if any(n > MAX_COUNT for n in counts.values()):
        return None
    return np.array([counts.get(e, 0) for e in ELEMENTS], np.int64)


def count_formula(counts):
    values = {e: int(n) for e, n in zip(ELEMENTS, counts) if n}
    order = (
        (["C", "H"] + sorted(set(values) - {"C", "H"}))
        if "C" in values
        else sorted(values)
    )
    return "".join(
        e + (str(values[e]) if values[e] > 1 else "") for e in order if e in values
    )


class FormulaPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, len(ELEMENTS) * (MAX_COUNT + 1)),
        )
        self.register_buffer("counts", torch.arange(MAX_COUNT + 1, dtype=torch.float32))

    def forward(self, z):
        return self.net(z).reshape(-1, len(ELEMENTS), MAX_COUNT + 1)

    def soft(self, z):
        return (self(z).softmax(-1) * self.counts).sum(-1) / 32.0

    @torch.inference_mode()
    def hypotheses(self, z, k=5):
        logits = self(z).log_softmax(-1)[0]
        beam = [([], 0.0)]
        for probabilities in logits:
            values, ids = probabilities.topk(min(k, len(probabilities)))
            possibilities = [
                (prefix + [int(i)], score + float(v))
                for prefix, score in beam
                for i, v in zip(ids, values)
            ]
            beam = sorted(possibilities, key=lambda x: -x[1])[:k]
        return [
            {
                "formula": count_formula(counts),
                "counts": counts,
                "log_probability": score,
            }
            for counts, score in beam
        ]


@torch.inference_mode()
def conditions(root, split):
    root = Path(root)
    directory = root / "generation_conditions_v2" / split
    directory.mkdir(parents=True, exist_ok=True)
    path = TRAIN if split == "train60k" else root / f"{split}.parquet"
    marker = directory / "complete.json"
    spec = {
        "version": 2,
        "input_sha256": digest(path),
        "encoder_sha256": digest(ENCODER),
        "neutral_mass_condition": True,
    }
    if marker.exists():
        if json.loads(marker.read_text()) != spec:
            raise ValueError("Generation condition cache changed")
        return directory
    model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if split == "train60k":
        features = Path("artifacts/scale_20260929/features/train60k")
    else:
        from casmi_ml.data import cache_features

        features = root / "features" / split
        if not (features / "complete.json").exists():
            cache_features(pd.read_parquet(path), checkpoint["preprocessing"], features)
    dataset = SelectedFeatures(features, model.input_names, target=False)
    output = []
    for batch in DataLoader(dataset, batch_size=256):
        x = torch.cat([batch["hist"], batch["meta"], batch["loss"]], -1).to(device)
        z = model.encoder(x).cpu().numpy()
        output.append(np.concatenate([z, batch["meta"].numpy()], 1))
    frame = pd.read_parquet(path)
    mass_values = [neutral_mass(r) for r in frame.to_dict("records")]
    mass_features = np.array(
        [[m / 1250 if m is not None else 0.0, float(m is None)] for m in mass_values],
        dtype=np.float32,
    )
    np.save(
        directory / "condition.npy",
        np.concatenate([np.concatenate(output), mass_features], axis=1).astype(
            np.float32
        ),
    )
    write_json(marker, spec)
    return directory


class GenerationDataset(Dataset):
    def __init__(self, frame, values, vocabulary, indexes=None):
        self.frame, self.values, self.vocab = frame, values, vocabulary
        indexes = np.arange(len(frame)) if indexes is None else indexes
        self.sequences = {}
        self.labels = {}
        self.indexes = []
        for i in indexes:
            sequence = vocabulary.encode(frame.iloc[i].normalized_smiles)
            label = formula_counts(frame.iloc[i].molecular_formula)
            if sequence is not None and label is not None:
                self.indexes.append(int(i))
                self.sequences[int(i)] = sequence
                self.labels[int(i)] = label
        self.excluded = len(indexes) - len(self.indexes)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        index = self.indexes[i]
        return (
            torch.from_numpy(np.array(self.values[index], copy=True)),
            torch.tensor(self.sequences[index]),
            torch.tensor(self.labels[index]),
        )


def collate(items):
    return (
        torch.stack([x[0] for x in items]),
        nn.utils.rnn.pad_sequence(
            [x[1] for x in items], batch_first=True, padding_value=0
        ),
        torch.stack([x[2] for x in items]),
    )


@torch.inference_mode()
def validation(decoder, formula, data, device):
    decoder.eval()
    formula.eval()
    loss, total, exact, correct = 0.0, 0, 0, 0
    for z, tokens, label in DataLoader(data, batch_size=64, collate_fn=collate):
        z, tokens, label = z.to(device), tokens.to(device), label.to(device)
        logits = decoder(tokens[:, :-1], torch.cat([z, formula.soft(z)], 1))
        loss += float(
            F.cross_entropy(
                logits.transpose(1, 2), tokens[:, 1:], ignore_index=0, reduction="sum"
            )
        )
        total += int((tokens[:, 1:] != 0).sum())
        pred = formula(z).argmax(-1)
        exact += int((pred == label).all(1).sum())
        correct += len(label)
    return {
        "token_nll": loss / max(total, 1),
        "formula_exact_top1": exact / max(correct, 1),
        "molecules_or_spectra": correct,
        "decoder_formula_source": "predicted_soft_counts",
    }


def train(root=ROOT, seed=42, epochs=30, seconds=86400):
    root = Path(root)
    prepare(root)
    configure(seed, threads=4)
    if seconds <= 0 or seconds > 86400:
        raise ValueError("Stage budget must be in (0,24h]")
    run = root / "generation" / f"smiles_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    spec = {
        "seed": seed,
        "epochs": epochs,
        "seconds": seconds,
        "layers": 6,
        "width": 256,
        "length": 256,
        "train_sha256": digest(TRAIN),
        "encoder_sha256": digest(ENCODER),
        "dev_sha256": digest(root / "researchdev.parquet"),
        "condition_source": "frozen60k+metadata+neutral_mass+predicted_soft_formula",
        "neutral_mass_condition": True,
        "lr": 3e-4,
    }
    freeze(run / "config.json", spec)
    if (run / "result.json").exists():
        return json.loads((run / "result.json").read_text())
    frame = pd.read_parquet(TRAIN)
    vocabulary = SmilesVocabulary.fit(frame.normalized_smiles.unique())
    write_json(run / "vocabulary.json", vocabulary.tokens)
    values = np.load(conditions(root, "train60k") / "condition.npy", mmap_mode="r")
    dev = pd.read_parquet(root / "researchdev.parquet")
    devvalues = np.load(
        conditions(root, "researchdev") / "condition.npy", mmap_mode="r"
    )
    # Fixed one query per molecule for likelihood checkpoint selection.
    ids = [int(v[0]) for v in dev.groupby("inchikey14", sort=True).indices.values()]
    development = GenerationDataset(dev, devvalues, vocabulary, ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    dim = values.shape[1]
    formula = FormulaPredictor(dim).to(device)
    decoder = SmilesDecoder(len(vocabulary.tokens), dim + len(ELEMENTS)).to(device)
    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(formula.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(seed)
    groups = frame.groupby("inchikey14", sort=True).indices
    history = []
    best = float("inf")
    stale = 0
    budget = StageBudget(root, "generation", run, seconds)
    start = time.monotonic()
    deadline = start + budget.allowance
    for epoch in range(1, epochs + 1):
        if time.monotonic() >= deadline:
            break
        indexes = [int(rng.choice(ids)) for ids in groups.values()]
        dataset = GenerationDataset(frame, values, vocabulary, indexes)
        decoder.train()
        formula.train()
        total, count = 0.0, 0
        for z, tokens, label in DataLoader(
            dataset, batch_size=64, shuffle=True, collate_fn=collate
        ):
            if time.monotonic() >= deadline:
                break
            z, tokens, label = z.to(device), tokens.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = decoder(tokens[:, :-1], torch.cat([z, formula.soft(z)], 1))
                structure_loss = F.cross_entropy(
                    logits.transpose(1, 2), tokens[:, 1:], ignore_index=0
                )
                formula_loss = F.cross_entropy(formula(z).transpose(1, 2), label)
                loss = structure_loss + 0.25 * formula_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite generation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + list(formula.parameters()), 1.0
            )
            optimizer.step()
            total += float(loss.detach())
            count += 1
        report = validation(decoder, formula, development, device)
        entry = {
            "epoch": epoch,
            "loss": total / max(count, 1),
            "validation": report,
            "excluded_training": dataset.excluded,
            "excluded_dev": development.excluded,
            "seconds": time.monotonic() - start,
        }
        history.append(entry)
        budget.checkpoint()
        write_json(run / "history.json", history)
        print(entry, flush=True)
        if report["token_nll"] < best:
            best = report["token_nll"]
            stale = 0
            torch.save(
                {
                    "decoder": {k: v.cpu() for k, v in decoder.state_dict().items()},
                    "formula": {k: v.cpu() for k, v in formula.state_dict().items()},
                    "config": spec,
                    "vocabulary": vocabulary.tokens,
                    "condition_dim": dim,
                    "report": report,
                },
                run / "model.pt",
            )
        else:
            stale += 1
        if epoch >= 8 and stale >= 5:
            break
    result = {
        "status": "complete" if history else "budget_before_epoch",
        "history": history,
        "best_dev_token_nll": best if history else None,
        "seconds": time.monotonic() - start,
        "performance_acceptance": "requires_generation_evaluation",
        "formula_oracle_used": False,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else None,
    }
    budget.close()
    write_json(run / "result.json", result)
    return result


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    vocabulary = SmilesVocabulary(checkpoint["vocabulary"][4:])
    if vocabulary.tokens != checkpoint["vocabulary"]:
        raise ValueError("Vocabulary order mismatch")
    decoder = SmilesDecoder(
        len(vocabulary.tokens), checkpoint["condition_dim"] + len(ELEMENTS)
    ).to(device)
    formula = FormulaPredictor(checkpoint["condition_dim"]).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    formula.load_state_dict(checkpoint["formula"])
    return decoder.eval(), formula.eval(), vocabulary


def validate_generated(
    sequences, logp, finished, vocabulary, mass, hypotheses, evidences
):
    candidates = {}
    valid = 0
    terminated = 0
    mass_matching = 0
    for sequence, prob, end in zip(sequences, logp, finished):
        if not bool(end):
            continue
        terminated += 1
        smiles = vocabulary.decode(sequence)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
            continue
        valid += 1
        exact = Descriptors.ExactMolWt(mol)
        if mass is None or abs(exact - mass) > max(mass * 35e-6, 0.006):
            continue
        mass_matching += 1
        key = Chem.MolToInchiKey(mol)[:14]
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        formula_support = max(
            (
                math.exp(h["log_probability"])
                for h in hypotheses
                if h["formula"] == formula
            ),
            default=0.0,
        )
        chemistry = score_candidate(evidences, smiles)["score"]
        candidate = {
            "key": key,
            "smiles": smiles,
            "formula": formula,
            "mass": exact,
            "log_probability": float(prob),
            "formula_support": formula_support,
            "chemical_score": chemistry,
        }
        if (
            key not in candidates
            or candidate["log_probability"] > candidates[key]["log_probability"]
        ):
            candidates[key] = candidate
    base = sorted(candidates, key=lambda k: (-candidates[k]["log_probability"], k))
    # Fixed evidence rank fusion, no truth-based filtering.
    if base:
        from casmi_ml.chemistry import rerank

        base = rerank(
            base,
            {},
            [],
            0.25,
            fragment_scores={k: candidates[k]["chemical_score"] for k in base},
        )
        base = rerank(
            base,
            {},
            [],
            0.25,
            fragment_scores={k: candidates[k]["formula_support"] for k in base},
        )
    return [candidates[k] for k in base], {
        "samples": len(sequences),
        "terminated": terminated,
        "valid": valid,
        "mass_matching": mass_matching,
        "unique_mass_matching": len(candidates),
    }


def generate(
    root=ROOT,
    checkpoint=None,
    split="researchdev",
    limit=None,
    samples=128,
    fragmenter=None,
    oracle_formula=False,
    stable_sampling=False,
):
    root = Path(root)
    if limit is not None and limit < 1:
        raise ValueError("Generation limit must be positive")
    if split == "researchholdout" and not (root / "joint_selection.json").exists():
        raise ValueError(
            "Freeze all development choices before generation holdout evaluation"
        )
    configure(42, threads=4)
    checkpoint = Path(checkpoint or root / "generation/smiles_42/model.pt")
    if not 1 <= samples <= 128:
        raise ValueError("Samples must be in [1,128]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, formula, vocabulary = load_model(checkpoint, device)
    frame = pd.read_parquet(root / f"{split}.parquet")
    if stable_sampling:
        encoder, encoder_checkpoint = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device)
    values = np.load(conditions(root, split) / "condition.npy", mmap_mode="r")
    groups = list(frame.groupby("inchikey14", sort=True).indices.items())
    groups = groups if limit is None else groups[:limit]
    checkpoint_suffix = (
        ""
        if checkpoint.resolve() == (root / "generation/smiles_42/model.pt").resolve()
        else "_" + digest(checkpoint)[:12]
    )
    out = (
        root
        / "generation"
        / f"{split}_samples{samples}_limit{limit or 'all'}{'_oracle' if oracle_formula else ''}{'_stable_v2' if stable_sampling else ''}{checkpoint_suffix}.json"
    )
    spec = {
        "checkpoint_sha256": digest(checkpoint),
        "split_sha256": digest(root / f"{split}.parquet"),
        "samples": samples,
        "limit": limit,
        "temperature": 0.8,
        "formula_oracle": oracle_formula,
        "fragmenter_sha256": fragmenter.sha256 if fragmenter else None,
    }
    if stable_sampling:
        spec["sampling"] = "spectrum_hash_v1_and_shared_group_forward_v2"
    freeze(out.with_suffix(".config.json"), spec)
    if out.exists():
        return json.loads(out.read_text())
    partial = out.with_suffix(".partial.json")
    previous = json.loads(partial.read_text()) if partial.exists() else None
    output = previous["rows"] if previous else []
    start = time.monotonic()
    generator = torch.Generator(device=device).manual_seed(42)
    if previous:
        generator.set_state(torch.tensor(previous["rng_state"], dtype=torch.uint8))
    for count, (key, ids) in enumerate(groups, 1):
        if count <= len(output):
            continue
        group = frame.iloc[ids]
        if stable_sampling:
            from casmi_ml.generation_sampling import sampling_seed

            generator = torch.Generator(device=device).manual_seed(sampling_seed(group))
        if stable_sampling:
            from casmi_ml.generation_sampling import condition_for_group

            z = condition_for_group(
                encoder, encoder_checkpoint["preprocessing"], group, device
            )
        else:
            z = torch.from_numpy(values[ids].mean(0, keepdims=True)).to(device)
        hypotheses = formula.hypotheses(z, k=5)
        with torch.no_grad():
            if oracle_formula:
                counts = formula_counts(group.molecular_formula.iloc[0])
                if counts is None:
                    raise ValueError("Unsupported oracle formula composition")
                formula_condition = (
                    torch.tensor(counts, dtype=torch.float32, device=device)[None]
                    / 32.0
                )
                hypotheses = [
                    {
                        "formula": group.molecular_formula.iloc[0],
                        "counts": counts.tolist(),
                        "log_probability": 0.0,
                    }
                ]
            else:
                formula_condition = formula.soft(z)
            condition = torch.cat([z, formula_condition], 1)
            sequences, logp, finished = decoder.generate(
                condition, samples, generator=generator
            )
        masses = [
            m for r in group.to_dict("records") if (m := neutral_mass(r)) is not None
        ]
        mass = float(np.median(masses)) if masses else None
        evidence = [extract_evidence(r) for r in group.to_dict("records")]
        candidates, stats = validate_generated(
            sequences.cpu().tolist(),
            logp.cpu().tolist(),
            finished.cpu().tolist(),
            vocabulary,
            mass,
            hypotheses,
            evidence,
        )
        fragment_status = []
        if fragmenter is not None and candidates:
            scores = {}
            for raw in group.to_dict("records"):
                result = fragmenter.score(
                    raw, {c["key"]: c["smiles"] for c in candidates[:100]}
                )
                fragment_status.append(result["status"])
                for candidate_key, score in result["scores"].items():
                    scores[candidate_key] = max(scores.get(candidate_key, 0.0), score)
            from casmi_ml.chemistry import rerank

            ranking = rerank(
                [c["key"] for c in candidates], {}, [], 0.25, fragment_scores=scores
            )
            by_key = {c["key"]: c for c in candidates}
            candidates = [by_key[k] for k in ranking]
        output.append(
            {
                "key": key,
                "candidates": candidates,
                "statistics": stats,
                "formula_hypotheses": hypotheses,
                "fragment_status": fragment_status,
                "oracle_formula": oracle_formula,
            }
        )
        if count % 10 == 0:
            write_json(
                partial,
                {"rows": output, "rng_state": generator.get_state().cpu().tolist()},
            )
            print("generated", count, "seconds", time.monotonic() - start, flush=True)
    write_json(out, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "generate"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=86400)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--split", default="researchdev", choices=["researchdev", "researchholdout"]
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--metfrag-jar", type=Path)
    parser.add_argument(
        "--oracle-formula",
        action="store_true",
        help="Separate ideal-condition diagnostic; never competition inference",
    )
    a = parser.parse_args()
    if a.stage == "train":
        train(a.root, a.seed, a.epochs, a.seconds)
    else:
        from casmi_ml.metfrag import MetFrag

        fragmenter = (
            MetFrag(a.metfrag_jar, a.root / "generation_metfrag_cache")
            if a.metfrag_jar
            else None
        )
        generate(
            a.root,
            a.checkpoint,
            a.split,
            a.limit,
            a.samples,
            fragmenter,
            a.oracle_formula,
        )


if __name__ == "__main__":
    main()
