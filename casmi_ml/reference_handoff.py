"""Private exact sparse reference handoff for downstream181 evidence scoring."""

from pathlib import Path

import pandas as pd
from scipy import sparse

from baseline import make_matrix
from casmi_ml.data import write_json


def save_reference(records, destination):
    destination = Path(destination)
    destination.mkdir(parents=True)
    chunks = [
        make_matrix([r[3] for r in records[i : i + 8192]])
        for i in range(0, len(records), 8192)
    ]
    matrix = sparse.vstack(chunks, format="csr") if chunks else make_matrix([])
    sparse.save_npz(destination / "spectra.npz", matrix)
    pd.DataFrame(
        [r[:3] for r in records], columns=["mass", "inchikey14", "normalized_smiles"]
    ).to_parquet(destination / "rows.parquet", index=False)
    write_json(
        destination / "complete.json",
        {
            "spectra": len(records),
            "scope": "Unlabeled-query mass-filtered actual training library; private handoff only",
            "query_labels_used": False,
        },
    )
