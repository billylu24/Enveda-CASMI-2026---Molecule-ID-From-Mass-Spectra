"""Training-only hydrogen-composition audit of connected diagnostic-ion hypotheses."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.connected_fragment_audit import TARGETS
from casmi_ml.data import write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.extended_diagnostic_rules import RULES
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze

AUDIT = Path("artifacts/research_loop/rounds/0339_connected_fragment_training_audit")
ION_H = (5, 7, 5, 5, 0, 2)


def component_hydrogen_offsets(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("One connected valid training molecule required")
    atoms = list(molecule.GetAtoms())
    adjacency = [[n.GetIdx() for n in a.GetNeighbors()] for a in atoms]
    symbols = [a.GetSymbol() for a in atoms]
    # Count H in the original graph; do not saturate radicals created by cutting.
    attached = [
        a.GetTotalNumHs(includeNeighbors=True) if a.GetAtomicNum() != 1 else 0
        for a in atoms
    ]
    total = Counter(s for s in symbols if s != "H")
    total_h = sum(attached)
    found = [set() for _ in ION_H]
    for bond in molecule.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        reached, stack = {a}, [a]
        while stack:
            at = stack.pop()
            for other in adjacency[at]:
                if (at == a and other == b) or (at == b and other == a):
                    continue
                if other not in reached:
                    reached.add(other)
                    stack.append(other)
        heavy = Counter(symbols[i] for i in reached if symbols[i] != "H")
        hydrogen = sum(attached[i] for i in reached)
        for part, h in ((heavy, hydrogen), (total - heavy, total_h - hydrogen)):
            composition = tuple(sorted(part.items()))
            for i in range(len(ION_H)):
                if composition == TARGETS[i]:
                    found[i].add(h - ION_H[i])
    return tuple(tuple(sorted(offsets)) for offsets in found)


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Hydrogen audit is local-only")
    output = Path(output)
    audit = json.loads((AUDIT / "report.json").read_text())
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    if (
        protocol["source_sha256"] != digest("casmi_ml/connected_fragment_audit.py")
        or protocol["training_sha256"] != digest(TRAIN)
        or protocol["negative_table_sha256"] != digest(HARD)
        or audit["private_support_sha256"]
        != digest(AUDIT / "private_connected_support.npy")
        or audit["train_development_overlap"] != 0
    ):
        raise ValueError("Isolated training support provenance changed")
    cached = json.loads((CACHE / "report.json").read_text())
    for name, sha in cached["private_input_sha256"].items():
        if digest(CACHE / name) != sha:
            raise ValueError("Original cached chemistry changed")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "audit_report_sha256": digest(AUDIT / "report.json"),
            "training_sha256": digest(TRAIN),
            "negative_table_sha256": digest(HARD),
            "rule": "All original60000 isolated representatives bound by339/305. For exactly one non-ring single-bond cut, retain original atom H counts (no cut-valence saturation), record all candidate component H minus frozen six observed ion target H. Separate exactH,anyH,delta -1/0/+1 availability; no model fitting, selected threshold, developer truth input or energy/charge mechanism claim. All original negative table and training observed channels unchanged.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    started = time.monotonic()
    train = pd.read_parquet(TRAIN)
    rows = pd.read_parquet(CACHE / "training_rows.parquet")
    data = train_data()
    if list(rows.inchikey14) != data["keys"]:
        raise ValueError("Original molecule order required")
    ids = rows.representative_row.to_numpy()
    if list(train.iloc[ids].inchikey14) != data["keys"]:
        raise ValueError("Original representative identities required")
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    values = []
    for start in range(0, len(ids), 1024):
        if c.stopped():
            return None
        values.extend(
            component_hydrogen_offsets(s)
            for s in train.iloc[ids[start : start + 1024]].normalized_smiles
        )
    available = np.asarray([[bool(v) for v in row] for row in values], bool)
    if not np.array_equal(
        available, np.load(AUDIT / "private_connected_support.npy")[:, :6]
    ):
        raise ValueError("Hydrogen enumeration must preserve original heavy support")
    exact = np.asarray([[0 in v for v in row] for row in values], bool)
    near = np.asarray(
        [[any(abs(d) <= 1 for d in v) for v in row] for row in values], bool
    )
    broad = np.load(CACHE / "candidate_support.npy")[:, :6].astype(bool)
    chemistry = np.load(CACHE / "mean_chemical.npy")[:, :18:3]
    from casmi_ml.strict_mass_negative_mask import admissible_negatives

    compatible = admissible_negatives(data["masses"], hard)
    channels = []
    for i, rule in enumerate(RULES[:6]):
        observed = chemistry[:, i] > 0
        use = compatible & observed[:, None]
        offsets = Counter(
            d
            for n, row in enumerate(values)
            if observed[n] and broad[n, i]
            for d in row[i]
        )
        contrasts = {}
        for name, support in (
            ("heavy_only", available),
            ("original_h_exact", exact),
            ("original_h_plus_minus1", near),
        ):
            refined = broad[:, i] & support[:, i]
            delta = refined[:, None].astype(np.int8) - refined[hard]
            contrasts[name] = {
                "positive_advantage": int((use & (delta > 0)).sum()),
                "negative_advantage": int((use & (delta < 0)).sum()),
                "ties": int((use & (delta == 0)).sum()),
                "observed_positive_supported": int((observed & refined).sum()),
            }
        channels.append(
            {
                "channel": rule.name,
                "observed_compatible_pairs": int(use.sum()),
                "observed_positive_offset_component_occurrences": dict(
                    sorted(offsets.items())
                ),
                "contrasts": contrasts,
            }
        )
    np.save(output / "private_exact_h_support.npy", exact)
    np.save(output / "private_near_h_support.npy", near)
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": 60000,
        "train_development_overlap": 0,
        "channels": channels,
        "seconds": time.monotonic() - started,
        "private_input_sha256": {
            n: digest(output / n)
            for n in ("private_exact_h_support.npy", "private_near_h_support.npy")
        },
        "scope": "Training-only available original-H component composition and association; no fitted scoring or validated mechanism. Fixed ±1 probe is a descriptive band, not chosen deployment tolerance. No development score or newly opened holdout.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
