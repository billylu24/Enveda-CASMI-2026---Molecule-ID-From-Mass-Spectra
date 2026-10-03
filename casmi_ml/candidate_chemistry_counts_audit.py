"""Training-only audit of broad SMARTS multiplicity before any new scoring model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.chemistry import RULES as BASE_RULES
from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import KINDS
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.extended_diagnostic_rules import RULES as EXTRA_RULES
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives
from casmi_ml.training_query_pairs import paired_rows

RULES = BASE_RULES + EXTRA_RULES
PATTERNS = tuple(Chem.MolFromSmarts(rule.smarts) for rule in RULES)


def motif_counts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Valid training molecule required")
    # Unique atom-set matches prevent query automorphisms from multiplying counts.
    counts = []
    for pattern in PATTERNS:
        matches = mol.GetSubstructMatches(pattern, uniquify=True, maxMatches=100000)
        if len(matches) >= 100000:
            raise ValueError("SMARTS multiplicity enumeration truncated")
        counts.append(len({frozenset(match) for match in matches}))
    return tuple(counts)


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Motif counts audit is local-only")
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "negative_table_sha256": digest(HARD),
            "chemistry_sha256": digest("casmi_ml/chemistry.py"),
            "extended_rule_sha256": digest("casmi_ml/extended_diagnostic_rules.py"),
            "cache_report_sha256": digest(CACHE / "report.json"),
            "rule": "All60000 original isolated training molecules, same pinned15 negatives and same paired-row binding. Count unique atom-set matches for unchanged20 broad SMARTS; compare binary aliases versus unbounded integer multiplicity on fixed mass-compatible subset. New36 observed chemistry from bound305 mean/sqrt cache; no fitting, dev labels, confidence tuning or absent-feature penalty.",
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
        raise ValueError("Original train-only identities required")
    rng = np.random.default_rng(42)
    ids = np.array([rng.choice(v) for v in groups.values()])
    partner = paired_rows(train, ids)
    cache_report = json.loads((CACHE / "report.json").read_text())
    cached_protocol = json.loads((CACHE / "protocol.json").read_text())
    for field, source in [
        ("training_sha256", TRAIN),
        ("negative_table_sha256", HARD),
        ("feature_sha256", "casmi_ml/extended_diagnostic_rules.py"),
        ("chemistry_sha256", "casmi_ml/chemistry.py"),
    ]:
        if cached_protocol[field] != digest(source):
            raise ValueError("Cached feature source changed")
    for name, sha in cache_report["private_input_sha256"].items():
        if digest(CACHE / name) != sha:
            raise ValueError("Cached training features changed")
    cached_rows = pd.read_parquet(CACHE / "training_rows.parquet")
    if (
        list(cached_rows.inchikey14) != data["keys"]
        or not np.array_equal(cached_rows.representative_row, ids)
        or not np.array_equal(cached_rows.paired_row, partner)
    ):
        raise ValueError("Paired-row cache binding differs")
    counts = []
    for start in range(0, len(ids), 1024):
        if c.stopped():
            return None
        counts.extend(
            motif_counts(s)
            for s in train.iloc[ids[start : start + 1024]].normalized_smiles
        )
    counts = np.asarray(counts, dtype=np.int32)
    support = counts > 0
    if not np.array_equal(
        support[:, len(BASE_RULES) :], np.load(CACHE / "candidate_support.npy")
    ):
        raise ValueError("Count presence must equal frozen SMARTS support")
    compatible = admissible_negatives(data["masses"], hard)
    binary_alias = (support[:, None, :] == support[hard]).all(-1)
    count_alias = (counts[:, None, :] == counts[hard]).all(-1)
    aliases = compatible & binary_alias
    chemistry = np.load(CACHE / "mean_chemical.npy")
    associations = []
    for i, rule in enumerate(EXTRA_RULES):
        col = len(BASE_RULES) + i
        pos = counts[:, col]
        neg = counts[hard, col]
        for j, kind in enumerate(KINDS):
            observed = chemistry[:, i * 3 + j] > 0
            use = compatible & observed[:, None]
            # Binary support tied but count differs; no directional score assumed.
            tied = use & (support[:, col, None] == support[hard, col])
            associations.append(
                {
                    "channel": rule.name + ":" + kind,
                    "observed_compatible_pairs": int(use.sum()),
                    "observed_binary_tied_pairs": int(tied.sum()),
                    "observed_binary_ties_separated_by_count": int(
                        (tied & (pos[:, None] != neg)).sum()
                    ),
                    "observed_positive_count_larger": int(
                        (use & (pos[:, None] > neg)).sum()
                    ),
                    "observed_negative_count_larger": int(
                        (use & (pos[:, None] < neg)).sum()
                    ),
                    "mean_positive_minus_negative_count": float(
                        (pos[:, None] - neg)[use].mean()
                    )
                    if use.any()
                    else None,
                }
            )
    np.save(output / "private_motif_counts.npy", counts)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "train_development_overlap": 0,
        "unique_binary_patterns": len(np.unique(support, axis=0)),
        "unique_count_patterns": len(np.unique(counts, axis=0)),
        "mass_compatible_pairs": int(compatible.sum()),
        "all20_binary_alias_pairs": int(aliases.sum()),
        "all20_count_alias_pairs": int((compatible & count_alias).sum()),
        "binary_aliases_separated_by_multiplicity": int((aliases & ~count_alias).sum()),
        "motifs": [
            {
                "name": rule.name,
                "maximum_count": int(counts[:, i].max()),
                "positive_molecules_with_multiple_matches": int(
                    (counts[:, i] > 1).sum()
                ),
            }
            for i, rule in enumerate(RULES)
        ],
        "observed_extended_channels": associations,
        "private_counts_sha256": digest(output / "private_motif_counts.npy"),
        "seconds": time.monotonic() - started,
        "scope": "Training-only multiplicity availability and association; count diversity is not predictive benefit or chemical fragmentation mechanism. No new fitted model or development score, no best replacement.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(p.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
