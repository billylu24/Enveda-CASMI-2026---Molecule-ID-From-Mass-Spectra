"""Training-only fixed multiplicity-capacity probe; retain a valid original60D model."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml.data import write_json
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.full_gap_candidate_training import run as prepare
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

COUNTS = Path("artifacts/research_loop/rounds/0332_candidate_chemistry_counts_audit")


def multiplicity_features(interactions, selected_counts):
    interactions = np.asarray(interactions)
    counts = np.asarray(selected_counts)
    if interactions.shape[-1] != 60 or counts.shape != (*interactions.shape[:-1], 20):
        raise ValueError("Original60 interactions and candidate20 counts must align")
    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("Finite nonnegative integral motif counts required")
    # One matched motif retains the original presence channel and zero extra.
    # Additional matches add a bounded magnitude; missing chemistry stays zero.
    extra_strength = np.maximum(counts - 1, 0) / (counts + 1)
    return interactions * np.repeat(extra_strength, 3, axis=-1)


def run(output):
    if not Controller().local_only():
        raise ValueError("Count capacity probe is local-only")
    output = Path(output)
    counts_report = json.loads((COUNTS / "report.json").read_text())
    if (
        digest(COUNTS / "private_motif_counts.npy")
        != counts_report["private_counts_sha256"]
    ):
        raise ValueError("Exact isolated training counts required")
    counts = np.load(COUNTS / "private_motif_counts.npy")
    hard = np.load(HARD)
    if counts.shape != (60000, 20) or hard.shape != (60000, 15):
        raise ValueError("Original60K motif counts/pinned15 required")
    selected = np.concatenate([np.arange(60000)[:, None], hard], axis=1)
    freeze(
        output / "audit_protocol.json",
        {
            "source_sha256": digest(__file__),
            "preparation_source_sha256": digest(
                "casmi_ml/full_gap_candidate_training.py"
            ),
            "solver_source_sha256": digest("casmi_ml/full_gap_candidate_bonus.py"),
            "count_report_sha256": digest(COUNTS / "report.json"),
            "private_counts_sha256": counts_report["private_counts_sha256"],
            "pinned_negative_sha256": digest(HARD),
            "rule": "Same327 original60K/120K mean/sqrt/cosine/pinned15. Compare original60D against original60 plus60 observed-chemistry times multiplicity extras: max(n-1,0)/(n+1). Count1 extra0, count0 chemistry interaction0. Same total nonnegative2.0 cap, CE scale, L2.01, SLSQP200/ftol1e-10. Training-only fixed hypothesis grounded in331/332 aliases. Keep valid original60D preparation model; private120D weights never passed to original runtime. No development score, fit, or activation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    observed = {}

    def capture(cosines, interactions, scale, mask=False, checkpoint=None):
        original, history = fit_bonus(cosines, interactions, scale, mask, checkpoint)
        extra = multiplicity_features(interactions, counts[selected])
        combined = np.concatenate([interactions, extra], axis=-1)
        new, enlarged = fit_bonus(cosines, combined, scale, mask, checkpoint)
        np.save(output / "private_count_capacity_weights.npy", new)
        observed.update(
            original=history,
            count_augmented=enlarged,
            augmented_active_channels=int((new > 1e-6).sum()),
            count_active_channels=int((new[60:] > 1e-6).sum()),
            extra_weight_sum=float(new[60:].sum()),
            training_objective_gain=history["final_objective"]
            - enlarged["final_objective"],
        )
        return original, history

    with patch("casmi_ml.full_gap_candidate_training.fit_bonus", new=capture):
        prepared = prepare(output)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "capacity_probe": observed,
        "original_preparation": prepared,
        "scope": "Training-only fixed120D hypothesis capacity; optimizer gain cannot qualify development accuracy. Original valid60D model retained, no augmented runtime or best activation.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(p.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
