"""Training-only discrimination and motif-aliasing audit of the finalized full-gap bonus."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.chemistry import RULES as BASE_RULES
from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import KINDS, evidence_features
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.extended_diagnostic_rules import RULES as EXTRA_RULES
from casmi_ml.full_gap_candidate_compatibility import load_model, structural_support
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives
from casmi_ml.training_query_pairs import paired_rows

CACHE = Path(
    "artifacts/research_loop/rounds/0305_extended_diagnostic_rule_training_audit"
)
MODEL = Path(
    "artifacts/research_loop/rounds/0327_full_gap_candidate_chemical/model.json"
)


def contrast_summary(positive, negative, compatible):
    positive, negative = np.asarray(positive), np.asarray(negative)
    compatible = np.asarray(compatible, dtype=bool)
    if (
        positive.ndim != 1
        or negative.shape != compatible.shape
        or negative.shape[0] != len(positive)
    ):
        raise ValueError("Aligned positive and pinned negative bonus arrays required")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Finite bonuses required")
    delta = positive[:, None] - negative
    report = {}
    for name, use in [
        ("unrestricted", np.ones_like(compatible)),
        ("mass_compatible", compatible),
    ]:
        selected = delta[use]
        report[name] = {
            "pairs": int(use.sum()),
            "positive_bonus_larger": int((selected > 1e-9).sum()),
            "negative_bonus_larger": int((selected < -1e-9).sum()),
            "bonus_ties": int((np.abs(selected) <= 1e-9).sum()),
            "mean_positive_minus_negative": float(selected.mean())
            if len(selected)
            else None,
            "molecules_with_any_positive_advantage": int(
                ((delta > 1e-9) & use).any(1).sum()
            ),
            "molecules_with_any_negative_advantage": int(
                ((delta < -1e-9) & use).any(1).sum()
            ),
        }
    return report


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Specificity audit is local-only")
    output = Path(output)
    _, model = load_model(MODEL, ENCODER, CRITIC)
    cache_report = json.loads((CACHE / "report.json").read_text())
    protocol = json.loads((CACHE / "protocol.json").read_text())
    for field, source in [
        ("training_sha256", TRAIN),
        ("negative_table_sha256", HARD),
        ("feature_sha256", "casmi_ml/extended_diagnostic_rules.py"),
        ("chemistry_sha256", "casmi_ml/chemistry.py"),
    ]:
        if protocol[field] != digest(source):
            raise ValueError("Training cache provenance changed: " + field)
    for name, sha in cache_report["private_input_sha256"].items():
        if digest(CACHE / name) != sha:
            raise ValueError("Training-only cached inputs changed")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "model_sha256": digest(MODEL),
            "training_sha256": digest(TRAIN),
            "negative_table_sha256": digest(HARD),
            "cache_report_sha256": digest(CACHE / "report.json"),
            "rule": "All60000 original training representatives+same-ion partners, original mean/sqrt60 channels and finalized327 weights; exact source-bound305 cache reuse for new36. Compare original pinned15 unrestricted and fixed35ppm/.006Da mass-compatible subset. No fitting or development truth/rank input.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    started = time.monotonic()
    train = pd.read_parquet(TRAIN)
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    groups = train.groupby("inchikey14", sort=True).indices
    development = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    if (
        list(groups) != data["keys"]
        or set(groups) & development
        or len(groups) != 60000
    ):
        raise ValueError("Original isolated training keys required")
    rng = np.random.default_rng(42)
    ids = np.array([rng.choice(v) for v in groups.values()])
    pair = paired_rows(train, ids)
    rows = pd.read_parquet(CACHE / "training_rows.parquet")
    if (
        list(rows.inchikey14) != data["keys"]
        or not np.array_equal(rows.representative_row, ids)
        or not np.array_equal(rows.paired_row, pair)
    ):
        raise ValueError("Cached representative/partner identities changed")
    positions = np.column_stack([ids, pair]).reshape(-1)
    features = []
    for start in range(0, len(positions), 1024):
        if c.stopped():
            return None
        features.extend(
            evidence_features(r, True)
            for r in train.iloc[positions[start : start + 1024]].to_dict("records")
        )
    original = np.stack(features).reshape(60000, 2, 24).mean(1)
    chemical = np.concatenate([original, np.load(CACHE / "mean_chemical.npy")], axis=1)
    support = np.asarray(
        [structural_support(s) for s in train.iloc[ids].normalized_smiles], dtype=bool
    )
    if not np.array_equal(
        support[:, len(BASE_RULES) :], np.load(CACHE / "candidate_support.npy")
    ):
        raise ValueError("Cached broad structural support differs")
    weighted = chemical * np.asarray(model["weights"], dtype=np.float64)
    positive = (weighted * np.repeat(support, 3, axis=1)).sum(1)
    negative = (weighted[:, None, :] * np.repeat(support[hard], 3, axis=-1)).sum(-1)
    compatible = admissible_negatives(data["masses"], hard)
    motifs_equal = (support[:, None, :] == support[hard]).all(-1)
    active = weighted > 1e-12
    weighted_support_equal = (
        ~active[:, None, :]
        | (
            np.repeat(support[:, None, :], 3, axis=-1)
            == np.repeat(support[hard], 3, axis=-1)
        )
    ).all(-1)
    contrasts = contrast_summary(positive, negative, compatible)
    for name, use in [
        ("unrestricted", np.ones_like(compatible)),
        ("mass_compatible", compatible),
    ]:
        contrasts[name].update(
            all20_motif_alias_pairs=int((motifs_equal & use).sum()),
            observed_weighted_channel_alias_pairs=int(
                (weighted_support_equal & use).sum()
            ),
        )
    channels = []
    for i, rule in enumerate(BASE_RULES + EXTRA_RULES):
        for j, kind in enumerate(KINDS):
            index = i * 3 + j
            observed = chemical[:, index] > 0
            eligible = observed[:, None] & compatible
            difference = chemical[:, index, None] * (
                support[:, i, None].astype(np.float32)
                - support[hard, i].astype(np.float32)
            )
            channels.append(
                {
                    "channel": rule.name + ":" + kind,
                    "weight": model["weights"][index],
                    "observed_training_molecules": int(observed.sum()),
                    "mass_compatible_observed_pairs": int(eligible.sum()),
                    "observed_positive_support_fraction": float(
                        support[observed, i].mean()
                    )
                    if observed.any()
                    else None,
                    "observed_negative_support_fraction": float(
                        support[hard, i][eligible].mean()
                    )
                    if eligible.any()
                    else None,
                    "mean_interaction_contrast_on_observed_compatible_pairs": float(
                        difference[eligible].mean()
                    )
                    if eligible.any()
                    else None,
                }
            )
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "training_slots": 120000,
        "train_development_overlap": 0,
        "weight_sum": float(np.sum(model["weights"])),
        "unique20_motif_patterns": len(np.unique(support, axis=0)),
        "positive_bonus_quantiles": dict(
            zip(
                ["q0", "q25", "q50", "q75", "q100"],
                map(float, np.quantile(positive, [0, 0.25, 0.5, 0.75, 1])),
            )
        ),
        "contrasts": contrasts,
        "channels": channels,
        "seconds": time.monotonic() - started,
        "scope": "Training-only broad-motif discrimination/aliasing audit, not full candidate pool recall, chemical mechanism proof, or accuracy qualification. No model/route/source changes or newly opened holdout.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(p.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
