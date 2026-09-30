"""Validated multi-spectrum consensus version of the CASMI library baseline."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import ADDUCT_MASS, load_candidates, make_matrix, vectorize


def predict_consensus(test, records, popularity):
    if not records:
        raise RuntimeError("No library candidates matched the test masses")
    masses = np.asarray([row[0] for row in records])
    keys = [row[1] for row in records]
    smiles = [row[2] for row in records]
    matrix = make_matrix([row[3] for row in records])
    order = np.argsort(masses)
    sorted_masses = masses[order]
    fallback = [pair for pair, _ in popularity.most_common(25)]
    predictions = {}
    for count, (molecule_id, group) in enumerate(test.groupby("molecule_id", sort=False)):
        neutral = np.asarray([row.precursor_mz - ADDUCT_MASS.get(row.adduct, math.nan)
                              for row in group.itertuples()])
        neutral = neutral[np.isfinite(neutral)]
        center = float(np.median(neutral)) if len(neutral) else 0.0
        delta = max(center * 35e-6, .006)
        start = np.searchsorted(sorted_masses, center - delta)
        stop = np.searchsorted(sorted_masses, center + delta, side="right")
        indexes = order[start:stop]
        if not len(indexes):
            nearest = np.searchsorted(sorted_masses, center)
            indexes = order[max(0, nearest - 150):nearest + 150]
        vectors = [vectorize(row.ms2_mzs, row.ms2_normalized_intensities)
                   for row in group.itertuples()]
        scores = (make_matrix(vectors) @ matrix[indexes].T).toarray()
        by_key = defaultdict(list)
        for col, index in enumerate(indexes):
            by_key[keys[index]].append(col)
        ranked = []
        for key, cols in by_key.items():
            per_query = scores[:, cols].max(axis=1)
            peak = float(per_query.max())
            combined = .65 * peak + .35 * float(np.mean(np.sort(per_query)[-3:]))
            best_col = cols[int(np.argmax(scores[:, cols].max(axis=0)))]
            ranked.append((combined, key, smiles[indexes[best_col]]))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        result = [row[2] for row in ranked[:25]]
        if not result:
            result = [pair[1] for pair in fallback]
        predictions[molecule_id] = ";".join(result)
        if count % 100 == 0:
            print(f"predicted {count + 1}/{test.molecule_id.nunique()} molecules", flush=True)
    return predictions


def main(data_dir, output):
    data_dir = Path(data_dir)
    if not (data_dir / "train.parquet").exists():
        matches = list(Path("/kaggle/input").rglob("train.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError("Competition train.parquet not mounted")
        data_dir = matches[0].parent
    test = pd.read_parquet(data_dir / "test.parquet")
    neutral = test.precursor_mz.to_numpy() - test.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral)]
    print(f"Test: {len(test):,} spectra, {test.molecule_id.nunique()} molecules", flush=True)
    records, popularity = load_candidates(data_dir / "train.parquet", neutral)
    predictions = predict_consensus(test, records, popularity)
    submission = test[["molecule_id"]].drop_duplicates().copy()
    submission["smiles"] = submission.molecule_id.map(predictions)
    if submission.smiles.isna().any() or submission.molecule_id.duplicated().any():
        raise ValueError("Incomplete predictions")
    if submission.smiles.str.count(";").max() >= 25:
        raise ValueError("More than 25 candidates")
    submission.to_csv(output, index=False)
    print(f"Wrote {output}: {len(submission)} molecules", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submission_v2.csv")
    args = parser.parse_args()
    main(args.data_dir, args.output)
