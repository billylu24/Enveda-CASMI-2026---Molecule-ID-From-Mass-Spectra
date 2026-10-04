"""Query-independent exact vectors derived only from the complete supplied train file."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from baseline import formula_mass, make_matrix, vectorize
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.reference_reuse import neutral_subset
from casmi_ml.research_protocol import freeze


def prepare(train, output):
    train, output = Path(train), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "baseline_sha256": digest("baseline.py"),
            "train_sha256": digest(train),
            "rule": "Scan entire supplied train with original formula_mass, spectrum quality filters, vectorize and make_matrix. Preserve original row order and float32 sparse values/indices. No target masses, query inputs, query scores or query predictions are used in preparation.",
            "query_dependent": False,
            "independent_acceptance": False,
        },
    )
    columns = [
        "molecular_formula",
        "normalized_smiles",
        "inchikey14",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_error_ppm",
    ]
    formulas, rows, chunks = {}, [], []
    started = time.monotonic()
    source = pq.ParquetFile(train)
    for number, batch in enumerate(
        source.iter_batches(batch_size=8192, columns=columns)
    ):
        vectors = []
        for formula, smiles, key, mzs, intensity, ppm in zip(
            *[batch.column(i).to_pylist() for i in range(6)]
        ):
            if formula not in formulas:
                formulas[formula] = formula_mass(formula)
            mass = formulas[formula]
            if (
                not np.isfinite(mass)
                or not smiles
                or not key
                or not mzs
                or not intensity
            ):
                continue
            if ppm is not None and abs(ppm) > 30:
                continue
            vector = vectorize(mzs, intensity)
            if vector:
                rows.append((mass, str(key), str(smiles)))
                vectors.append(vector)
        chunks.append(make_matrix(vectors))
        if number % 50 == 0:
            print("train vectors", number * 8192, "usable", len(rows), flush=True)
    matrix = sparse.vstack(chunks, format="csr") if chunks else make_matrix([])
    if matrix.shape[0] != len(rows):
        raise ValueError("Complete train row/vector alignment required")
    sparse.save_npz(output / "spectra.npz", matrix)
    pd.DataFrame(rows, columns=["mass", "inchikey14", "normalized_smiles"]).to_parquet(
        output / "rows.parquet", index=False
    )
    report = {
        "valid": True,
        "query_dependent": False,
        "source_rows": source.metadata.num_rows,
        "usable_spectra": len(rows),
        "nonzero": matrix.nnz,
        "seconds": time.monotonic() - started,
        "source_sha256": digest(__file__),
        "baseline_sha256": digest("baseline.py"),
        "train_sha256": digest(train),
        "rows_sha256": digest(output / "rows.parquet"),
        "spectra_sha256": digest(output / "spectra.npz"),
        "scope": "All train rows, original exact formula/vector/quality rules; zero query inputs or predictions. Must separately prove subset rows and every sparse entry equal original reference construction before runtime use.",
    }
    write_json(output / "report.json", report)
    return report


def select_rows(masses, targets):
    targets = np.sort(np.asarray(targets, np.float64))
    if not len(targets):
        raise ValueError("Nonempty reference mass targets required")
    masses = np.asarray(masses, np.float64)
    positions = np.searchsorted(targets, masses)
    left = targets[np.maximum(positions - 1, 0)]
    right = targets[np.minimum(positions, len(targets) - 1)]
    # Preserve original left-to-right arithmetic, including boundary rounding.
    return np.flatnonzero(
        np.isfinite(masses)
        & (np.minimum(abs(masses - left), abs(masses - right)) <= masses * 35 * 1e-6)
    )


def build_cached_reference(train, neutral, centers, destination, *, store):
    store, destination = Path(store), Path(destination)
    report = json.loads((store / "report.json").read_text())
    if (
        not report["valid"]
        or report["query_dependent"]
        or report["source_sha256"] != digest(__file__)
        or report["baseline_sha256"] != digest("baseline.py")
        or report["train_sha256"] != digest(train)
    ):
        raise ValueError("Complete original train/vector source binding required")
    for name, field in [
        ("rows.parquet", "rows_sha256"),
        ("spectra.npz", "spectra_sha256"),
    ]:
        if digest(store / name) != report[field]:
            raise ValueError("Train-only vector cache asset changed")
    rows = pd.read_parquet(store / "rows.parquet")
    matrix = sparse.load_npz(store / "spectra.npz")
    if (
        matrix.shape[0] != len(rows)
        or len(rows) != report["usable_spectra"]
        or matrix.nnz != report["nonzero"]
    ):
        raise ValueError("Train reference dimensions differ")
    ids = select_rows(rows.mass.to_numpy(), sorted(set(neutral) | set(centers)))
    rows = rows.iloc[ids].reset_index(drop=True)
    matrix = matrix[ids]
    records = list(rows.itertuples(index=False, name=None))
    subset = neutral_subset(records, neutral)
    destination.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(destination / "rows.parquet", index=False)
    sparse.save_npz(destination / "spectra.npz", matrix)
    write_json(
        destination / "complete.json",
        {
            "spectra": len(rows),
            "query_labels_used": False,
            "scope": "Same mass-filtered reference from source-bound query-independent complete training vectors.",
        },
    )
    return [records[i] for i in subset], matrix[subset], set(rows.inchikey14)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--output", required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
