"""Train-only representation limits of exactly identical pinned fingerprint pairs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from casmi_ml.data import fingerprint, write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives


def representation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Valid actual training structure required")
    Chem.RemoveStereochemistry(mol)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=8192)
    return {
        "canonical_graph": Chem.MolToSmiles(mol, canonical=True),
        "unfolded_radius2_counts": generator.GetSparseCountFingerprint(
            mol
        ).GetNonzeroElements(),
        "bits4096": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        .GetFingerprint(mol)
        .ToBitString(),
        "bits8192": generator.GetFingerprint(mol).ToBitString(),
    }


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Fingerprint alias audit is local-only")
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "hard_table_sha256": digest(HARD),
            "fingerprint_source_sha256": digest("casmi_ml/data.py"),
            "rule": "All original60000 training molecules and fixed pinned15. Audit ONLY exactly identical original2048 radius2 bit vectors. Distinguish identical actual canonical non-stereo graph from bit folding and radius2 presence/count aliasing. Check4096/8192 bit and unfolded radius2 counts for these same pairs. No fitting, development predictions, labels, ranks or model activation. Mass subset uses original35ppm/.006Da.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    rows = pd.read_parquet(TRAIN, columns=["inchikey14", "normalized_smiles"])
    rows = rows.groupby("inchikey14", sort=True).first()
    devkeys = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    if list(rows.index) != data["keys"] or set(rows.index) & devkeys:
        raise ValueError("Original isolated training identity order required")
    packed = np.packbits(np.asarray(data["fps"], np.uint8), axis=1)
    pairs = []
    for start in range(0, len(packed), 256):
        if c.stopped():
            return None
        matches = (
            packed[start : start + 256, None] == packed[hard[start : start + 256]]
        ).all(-1)
        for row, slot in np.argwhere(matches):
            pairs.append(
                (start + int(row), int(hard[start + int(row), slot]), int(slot))
            )
    ids = sorted({i for p in pairs for i in p[:2]})
    smiles = rows.normalized_smiles.tolist()
    for i in ids:
        if not np.array_equal(fingerprint(smiles[i]), data["fps"][i]):
            raise ValueError("Actual training SMILES differs from original fingerprint")
    decoded = {i: representation(smiles[i]) for i in ids}
    compatible = admissible_negatives(data["masses"], hard)
    counts = {name: Counter() for name in ("unrestricted", "mass_compatible")}
    for left, right, slot in pairs:
        a, b = decoded[left], decoded[right]
        same = a["canonical_graph"] == b["canonical_graph"]
        for name in ("unrestricted", "mass_compatible"):
            if name == "mass_compatible" and not compatible[left, slot]:
                continue
            counts[name]["identical_original2048_pairs"] += 1
            counts[name][
                "same_actual_nonstereo_graph" if same else "distinct_actual_graph"
            ] += 1
            for field in ("bits4096", "bits8192", "unfolded_radius2_counts"):
                if a[field] != b[field]:
                    counts[name][field + "_separated"] += 1
            if (
                not same
                and a["unfolded_radius2_counts"] == b["unfolded_radius2_counts"]
            ):
                counts[name]["distinct_graph_unfolded_radius2_count_alias"] += 1
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": len(packed),
        "train_development_overlap": 0,
        "total_pinned_pairs": int(hard.size),
        "unique_alias_molecules": len(ids),
        "contrasts": {name: dict(v) for name, v in counts.items()},
        "scope": "Exact original fingerprint training aliases only, not all candidate-pool collisions or accuracy upper bound. Canonical non-stereo graph equality is not a verdict about supplied identity labels. A larger bit/count/graph representation needs matched training and full system qualification before activation.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
