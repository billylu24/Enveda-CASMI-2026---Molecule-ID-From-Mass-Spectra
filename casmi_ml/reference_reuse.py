"""Build the same legacy and union references from one mass-filtered scan."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from baseline import load_candidates, make_matrix
from casmi_ml.data import write_json


def reference_matrix(records):
    chunks = [
        make_matrix([r[3] for r in records[i : i + 8192]])
        for i in range(0, len(records), 8192)
    ]
    return sparse.vstack(chunks, format="csr") if chunks else make_matrix([])


def neutral_subset(records, neutral):
    targets = np.sort(np.asarray(neutral, dtype=np.float64))
    if not len(targets):
        raise ValueError("Nonempty legacy neutral masses required")
    masses = np.asarray([r[0] for r in records], dtype=np.float64)
    positions = np.searchsorted(targets, masses)
    left = targets[np.maximum(positions - 1, 0)]
    right = targets[np.minimum(positions, len(targets) - 1)]
    return np.flatnonzero(
        np.isfinite(masses)
        & (np.minimum(abs(masses - left), abs(masses - right)) <= masses * 35e-6)
    )


def build_shared_reference(train, neutral, centers, destination):
    records, _ = load_candidates(train, np.asarray(sorted(set(neutral) | set(centers))))
    matrix = reference_matrix(records)
    subset = neutral_subset(records, neutral)
    rows = [r[:3] for r in records]
    destination = Path(destination)
    destination.mkdir(parents=True)
    sparse.save_npz(destination / "spectra.npz", matrix)
    pd.DataFrame(rows, columns=["mass", "inchikey14", "normalized_smiles"]).to_parquet(
        destination / "rows.parquet", index=False
    )
    write_json(
        destination / "complete.json",
        {
            "spectra": len(rows),
            "query_labels_used": False,
            "scope": "Same charge-aware union reference, one scan; private runtime handoff",
        },
    )
    return [rows[i] for i in subset], matrix[subset], {r[1] for r in rows}
