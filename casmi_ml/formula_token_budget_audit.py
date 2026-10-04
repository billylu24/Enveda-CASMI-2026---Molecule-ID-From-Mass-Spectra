"""Check experimental heavy-token accounting against every training representative."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import write_json
from casmi_ml.formula_budget_sampling import atom_count_tokens
from casmi_ml.generation_experiment import formula_counts
from casmi_ml.generation_temperature_experiment import CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_models import SmilesVocabulary
from casmi_ml.research_protocol import TRAIN, freeze


def run(output):
    if not Controller().local_only():
        raise ValueError("Local-only training accounting required")
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "accounting_source_sha256": digest("casmi_ml/formula_budget_sampling.py"),
            "decoder_sha256": digest(CHECKPOINT),
            "training_sha256": digest(TRAIN),
            "rule": "All60000 training representative first rows, original frozen vocabulary/length limit; count supported explicit heavy atoms in entire encoded SMILES vs first recorded molecular formula excluding allhydrogen. Keep full denominator, separate unsupported formula/sequence/atom and exact/mismatch counts. No generation/dev/test labels or settings selection. Model unchanged; checks accounting feasibility, not structural recall.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    vocab = SmilesVocabulary(saved["vocabulary"])
    matrix, unsupported = atom_count_tokens(vocab)
    rows = (
        pd.read_parquet(
            TRAIN, columns=["inchikey14", "molecular_formula", "normalized_smiles"]
        )
        .groupby("inchikey14", sort=True)
        .first()
    )
    counts = Counter()
    mismatches = []
    for key, row in rows.iterrows():
        counts["molecules"] += 1
        target = formula_counts(row.molecular_formula)
        sequence = vocab.encode(
            row.normalized_smiles,
            limit=saved["decoder_config"]["limit"]
            if "decoder_config" in saved
            else 256,
        )
        if target is None:
            counts["unsupported_formula"] += 1
            continue
        if sequence is None:
            counts["unencodable_sequence"] += 1
            continue
        if unsupported[sequence].any():
            counts["unsupported_atom_tokens"] += 1
            continue
        target = target.copy()
        target[1] = 0
        observed = matrix[sequence].sum(0)
        if np.array_equal(target, observed):
            counts["exact_heavy_atom_accounting"] += 1
        else:
            counts["mismatched_heavy_atom_accounting"] += 1
            mismatches.append(key)
    if (
        counts["molecules"] != 60000
        or sum(value for name, value in counts.items() if name != "molecules") != 60000
    ):
        raise ValueError("Complete original60000 denominator required")
    write_json(output / "private_mismatch_keys.json", mismatches)
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "counts": dict(counts),
        "scope": "Training heavy-atom token accounting feasibility only. No guarantee of grammar validity/implicithydrogen/exact mass/graphrecall; no query oracle target or deployment. Unsupported cases must remain original unconstrained trajectories, finaloriginal RDKit validation mandatory.",
    }
    write_json(output / "report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
