"""Fit-only formula prior and mass-window proposals on a fixed molecular holdout."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from casmi_ml.chemistry import composition_mass, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.diagnostic_bonus_generalization_audit import molecular_bootstrap_gain
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.generation_experiment import (
    ELEMENTS,
    FormulaPredictor,
    conditions,
    count_formula,
    formula_counts,
)
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.training import configure


def joint_beam(logp, k=5):
    """Original independent-count top-k beam; no observed mass or labels."""
    values = np.asarray(logp)
    if values.shape != (len(ELEMENTS), 257) or not np.isfinite(values).all():
        raise ValueError(
            "Finite original fourteen-element count log probabilities required"
        )
    beam = [([], 0.0)]
    for row in values:
        ids = np.argsort(-row, kind="stable")[:k]
        beam = sorted(
            (
                (prefix + [int(i)], score + float(row[i]))
                for prefix, score in beam
                for i in ids
            ),
            key=lambda item: -item[1],
        )[:k]
    return [count_formula(counts) for counts, _ in beam]


def mass_proposals(logp, center, catalog_masses, catalog_counts, catalog_formulas, k=5):
    """Rank all fit-only empirical formulas in the original observed mass window."""
    if center is None or not np.isfinite(center) or center <= 0:
        return [], 0
    tolerance = max(center * 35e-6, 0.006)
    start = np.searchsorted(catalog_masses, center - tolerance, side="left")
    end = np.searchsorted(catalog_masses, center + tolerance, side="right")
    counts = catalog_counts[start:end]
    if not len(counts):
        return [], 0
    scores = np.asarray(logp)[np.arange(len(ELEMENTS))[None, :], counts].sum(1)
    ids = np.argsort(-scores, kind="stable")[:k]
    return [catalog_formulas[start + int(i)] for i in ids], len(counts)


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU training-internal formula diagnostic required")
    configure(42, threads=1)
    output = Path(output)
    train = pd.read_parquet(
        TRAIN, columns=["inchikey14", "molecular_formula", "adduct", "precursor_mz"]
    )
    groups = train.groupby("inchikey14", sort=True)
    keys = list(groups.indices)
    holdout = heldout_mask(keys)
    if len(keys) != 60000 or int(holdout.sum()) != 12006:
        raise ValueError("Fixed original60000/12006 molecular split required")
    cache = conditions(ROOT, "train60k")
    source = np.load(cache / "condition.npy", mmap_mode="r")
    xs, formulas, centers = [], [], []
    multiple_recorded_formulas = []
    for key in keys:
        ids = groups.indices[key]
        group = train.iloc[ids]
        # Same deployment reduction: mean all observable condition rows.
        xs.append(np.asarray(source[ids]).mean(0))
        values = group.molecular_formula.unique()
        multiple_recorded_formulas.append(len(values) != 1)
        # Preserve the original representative (first recorded row) label and
        # keep heterogeneous identities in both fit and heldout denominators.
        formulas.append(group.molecular_formula.iloc[0])
        masses = [
            m
            for row in group.drop(columns=["inchikey14", "molecular_formula"]).to_dict(
                "records"
            )
            if (m := neutral_mass(row)) is not None
        ]
        centers.append(float(np.median(masses)) if masses else None)
    x = np.asarray(xs, np.float32)
    parsed = [formula_counts(f) for f in formulas]
    supported = np.asarray([v is not None for v in parsed])
    labels = np.asarray(
        [v if v is not None else np.zeros(len(ELEMENTS), np.int64) for v in parsed]
    )
    fit = np.flatnonzero(~holdout & supported)
    held = np.flatnonzero(holdout)
    empirical = sorted(
        {count_formula(labels[i]) for i in fit}, key=lambda f: (composition_mass(f), f)
    )
    empirical_set = set(empirical)
    cat_counts = np.stack([formula_counts(f) for f in empirical])
    cat_masses = np.asarray([composition_mass(f) for f in empirical])
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "train_sha256": digest(TRAIN),
            "encoder_sha256": digest(ENCODER),
            "condition_sha256": digest(cache / "condition.npy"),
            "condition_config_sha256": digest(cache / "complete.json"),
            "split_source_sha256": digest(
                "casmi_ml/forward_diagnostic_training_audit.py"
            ),
            "seed": 42,
            "epochs": 12,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "rule": "Fresh random formula head, fit only original47994 molecule side supported element labels; fixed12 epochs, no holdout-selected checkpoint. Existing encoder pretrained on all60K, NOT independent end-to-end generalization. Mean original observable rows per molecule; original mass median/35ppm or.006Da. The first recorded row provides the representative formula label; retain and count identities with multiple recorded formulas, no exclusion. Candidate formulas ONLY unique fit-side representative empirical formulas, no holdout/dev labels/catalog. Compare unchanged original independent element top5 beam with all fit-catalog mass-window formulas ranked by the SAME head count log probabilities. Formula labels used only training fit or final aggregate metrics, never query proposal inputs. Retain all12006 holdout identities including unsupported/no-mass/no-candidates. No decoder training, structure inference, deployment, dev tuning or accuracy claim.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    device = torch.device("cuda")
    model = FormulaPredictor(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = np.random.default_rng(42)
    start = time.monotonic()
    budget = StageBudget(output, "generation", str(output), 1800)
    try:
        history = []
        for epoch in range(12):
            order = rng.permutation(fit)
            model.train()
            total = 0.0
            for offset in range(0, len(order), 256):
                if c.stopped() or not budget.checkpoint():
                    raise RuntimeError(
                        "Formula audit stopped before fixed training completed"
                    )
                ids = order[offset : offset + 256]
                logits = model(torch.from_numpy(x[ids]).to(device))
                loss = F.cross_entropy(
                    logits.transpose(1, 2), torch.from_numpy(labels[ids]).to(device)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            history.append({"epoch": epoch + 1, "fit_count_ce": total / len(fit)})
            print(history[-1], flush=True)
        torch.save(
            {"formula": model.cpu().state_dict(), "condition_dim": x.shape[1]},
            output / "private_formula.pt",
        )
        model.to(device).eval()
        (
            baseline_hits,
            projected_hits,
            mass_support,
            proposal_counts,
            true_in_fit,
            ce,
        ) = [], [], [], [], [], []
        with torch.inference_mode():
            for offset in range(0, len(held), 128):
                if c.stopped() or not budget.checkpoint():
                    raise RuntimeError("Formula audit stopped before complete holdout")
                ids = held[offset : offset + 128]
                logp = (
                    model(torch.from_numpy(x[ids]).to(device))
                    .log_softmax(-1)
                    .cpu()
                    .numpy()
                )
                for i, probabilities in zip(ids, logp):
                    base = joint_beam(probabilities)
                    proposals, n = mass_proposals(
                        probabilities, centers[i], cat_masses, cat_counts, empirical
                    )
                    truth = count_formula(labels[i]) if supported[i] else None
                    baseline_hits.append(bool(truth and truth in base))
                    projected_hits.append(bool(truth and truth in proposals))
                    mass_support.append(
                        bool(
                            truth
                            and centers[i] is not None
                            and abs(composition_mass(truth) - centers[i])
                            <= max(centers[i] * 35e-6, 0.006)
                        )
                    )
                    proposal_counts.append(n)
                    true_in_fit.append(bool(truth and truth in empirical_set))
                    ce.append(
                        float(
                            -probabilities[np.arange(len(ELEMENTS)), labels[i]].mean()
                        )
                        if supported[i]
                        else None
                    )
        baseline_hits, projected_hits = (
            np.asarray(baseline_hits),
            np.asarray(projected_hits),
        )
        # Private arrays preserve all12006 denominator and allow reproducibility.
        np.savez_compressed(
            output / "private_holdout_observations.npz",
            baseline_top5=baseline_hits,
            projected_top5=projected_hits,
            proposal_counts=proposal_counts,
            true_formula_in_fit=true_in_fit,
            observed_mass_compatible=mass_support,
        )
        report = {
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "molecules": 60000,
            "multiple_recorded_formula_molecules": int(sum(multiple_recorded_formulas)),
            "heldout_multiple_recorded_formula_molecules": int(
                np.asarray(multiple_recorded_formulas)[holdout].sum()
            ),
            "fit_supported_molecules": len(fit),
            "heldout_molecules": len(held),
            "heldout_supported_molecules": int(supported[held].sum()),
            "empirical_fit_formulas": len(empirical),
            "heldout_truth_formula_in_fit_catalog": int(sum(true_in_fit)),
            "heldout_truth_formula_in_observed_mass_window": int(sum(mass_support)),
            "heldout_truth_available_in_empirical_mass_window": int(
                sum(a and b for a, b in zip(true_in_fit, mass_support))
            ),
            "heldout_queries_with_empirical_mass_proposals": int(
                np.count_nonzero(proposal_counts)
            ),
            "proposal_count_quantiles": np.quantile(
                proposal_counts, [0, 0.5, 0.9, 0.99, 1]
            ).tolist(),
            "unconstrained_top5_hits": int(baseline_hits.sum()),
            "mass_projected_top5_hits": int(projected_hits.sum()),
            "improvements": int((projected_hits & ~baseline_hits).sum()),
            "regressions": int((baseline_hits & ~projected_hits).sum()),
            "paired_formula_recall_gain": molecular_bootstrap_gain(
                1 - baseline_hits.astype(float), 1 - projected_hits.astype(float)
            ),
            "mean_supported_count_nll": float(
                np.mean([v for v in ce if v is not None])
            ),
            "epochs": len(history),
            "fit_history": history,
            "seconds": time.monotonic() - start,
            "scope": "Training-internal formula proposal recall only. Empirical formula catalog upper bound and all60K-pretrained encoder limit generalization interpretation. No structure or complete ranking accuracy acceptance; fresh formula model private, not activated. Missing empirical formulas cannot be recovered by this search.",
        }
        write_json(output / "report.json", report)
        print(json.dumps(report), flush=True)
        return report
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
