"""Mass-filtered spectral-library search baseline for Enveda CASMI 2026.

Run locally: python baseline.py --data-dir data --output submission.csv
The same source is embedded in the Kaggle notebook for a code submission.
"""

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse


ATOMIC_MASS = {
    "H": 1.00782503223, "C": 12.0, "N": 14.00307400443,
    "O": 15.99491461957, "P": 30.97376199842, "S": 31.9720711744,
    "F": 18.99840316273, "Cl": 34.968852682, "Br": 78.9183376,
    "I": 126.904468, "Si": 27.97692653465, "B": 11.00930536,
    "Na": 22.9897692820, "K": 38.9637064864,
}
PROTON = 1.007276466621
WATER = 18.01056468403
ADDUCT_MASS = {
    "[M+H]+": PROTON,
    "[M+NH4]+": 18.033825553,
    "[M-H2O+H]+": PROTON - WATER,
    "[M-2H2O+H]+": PROTON - 2 * WATER,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -WATER - PROTON,
    "[M+CH2O2-H]-": 46.005479303 - PROTON,
    "[M+Cl]-": 34.969402,
}
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula):
    if not isinstance(formula, str) or not formula:
        return math.nan
    pos = 0
    mass = 0.0
    for match in FORMULA_PATTERN.finditer(formula):
        if match.start() != pos or match.group(1) not in ATOMIC_MASS:
            return math.nan
        mass += ATOMIC_MASS[match.group(1)] * int(match.group(2) or 1)
        pos = match.end()
    return mass if pos == len(formula) else math.nan


def vectorize(mzs, intensities, max_peaks=100):
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    valid = np.isfinite(mzs) & np.isfinite(intensities) & (intensities >= .01)
    valid &= (mzs >= 40) & (mzs < 1250)
    mzs, intensities = mzs[valid], intensities[valid]
    if len(mzs) > max_peaks:
        top = np.argpartition(intensities, -max_peaks)[-max_peaks:]
        mzs, intensities = mzs[top], intensities[top]
    bins = np.rint(mzs * 10).astype(np.int32)
    weights = np.sqrt(intensities) * np.sqrt(mzs / 100)
    by_bin = defaultdict(float)
    for bin_id, weight in zip(bins, weights):
        by_bin[int(bin_id)] = max(by_bin[int(bin_id)], float(weight))
    norm = math.sqrt(sum(weight * weight for weight in by_bin.values()))
    return {bin_id: weight / norm for bin_id, weight in by_bin.items()} if norm else {}


def make_matrix(vectors):
    rows, cols, vals = [], [], []
    for row, vector in enumerate(vectors):
        for col, value in vector.items():
            rows.append(row)
            cols.append(col)
            vals.append(value)
    return sparse.csr_matrix((np.float32(vals), (rows, cols)),
                             shape=(len(vectors), 12501), dtype=np.float32)


def load_candidates(train_path, target_masses, tolerance_ppm=35):
    """Scan once and materialize spectra only near a test neutral mass."""
    target_masses = np.sort(np.asarray(target_masses, dtype=np.float64))
    parquet = pq.ParquetFile(train_path)
    columns = ["molecular_formula", "normalized_smiles", "inchikey14",
               "ms2_mzs", "ms2_normalized_intensities", "precursor_error_ppm"]
    formula_cache = {}
    records = []
    popularity = Counter()
    for batch_no, batch in enumerate(parquet.iter_batches(batch_size=8192, columns=columns)):
        formulas = batch.column(0).to_pylist()
        for f in set(formulas):
            if f not in formula_cache:
                formula_cache[f] = formula_mass(f)
        masses = np.asarray([formula_cache[f] for f in formulas], dtype=np.float64)
        positions = np.searchsorted(target_masses, masses)
        left = target_masses[np.maximum(positions - 1, 0)]
        right = target_masses[np.minimum(positions, len(target_masses) - 1)]
        distance = np.minimum(abs(masses - left), abs(masses - right))
        keep = np.flatnonzero(np.isfinite(masses) &
                              (distance <= masses * tolerance_ppm * 1e-6))
        if len(keep):
            selected = batch.take(keep)
            for f, smiles, key, mzs, intensities, ppm in zip(*[
                selected.column(i).to_pylist() for i in range(6)
            ]):
                if not smiles or not key or not mzs or not intensities:
                    continue
                if ppm is not None and abs(ppm) > 30:
                    continue
                vector = vectorize(mzs, intensities)
                if vector:
                    records.append((formula_cache[f], str(key), str(smiles), vector))
                    popularity[(str(key), str(smiles))] += 1
        if batch_no % 50 == 0:
            print(f"scanned {batch_no * 8192:,}/{parquet.metadata.num_rows:,}; "
                  f"kept {len(records):,}", flush=True)
    print(f"Library spectra kept: {len(records):,}", flush=True)
    return records, popularity


def predict(test, records, popularity):
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
        neutral = np.asarray([
            row.precursor_mz - ADDUCT_MASS.get(row.adduct, math.nan)
            for row in group.itertuples()
        ])
        neutral = neutral[np.isfinite(neutral)]
        if not len(neutral):
            neutral = np.array([0.0])
        center = float(np.median(neutral))
        # 35 ppm accommodates modest instrument and adduct rounding errors.
        delta = max(center * 35e-6, .006)
        start = np.searchsorted(sorted_masses, center - delta, side="left")
        stop = np.searchsorted(sorted_masses, center + delta, side="right")
        indexes = order[start:stop]
        if not len(indexes):
            nearest = np.searchsorted(sorted_masses, center)
            indexes = order[max(0, nearest - 150):nearest + 150]
        query_vectors = [vectorize(row.ms2_mzs, row.ms2_normalized_intensities)
                         for row in group.itertuples()]
        query_vectors = [v for v in query_vectors if v]
        rank = {}
        if query_vectors and len(indexes):
            scores = (make_matrix(query_vectors) @ matrix[indexes].T).toarray().max(axis=0)
            for index, score in zip(indexes, scores):
                key = keys[index]
                if key not in rank or score > rank[key][0]:
                    rank[key] = (float(score), smiles[index])
        ordered = sorted(rank.items(), key=lambda item: item[1][0], reverse=True)
        result = [value[1] for _, value in ordered[:25]]
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
            contents = [str(p) for p in Path("/kaggle/input").glob("*")]
            raise FileNotFoundError(f"Cannot locate train.parquet in {data_dir} or /kaggle/input; top-level: {contents}")
        data_dir = matches[0].parent
    test = pd.read_parquet(data_dir / "test.parquet")
    neutral = test.precursor_mz.to_numpy() - test.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral)]
    print(f"Test: {len(test):,} spectra, {test.molecule_id.nunique()} molecules", flush=True)
    records, popularity = load_candidates(data_dir / "train.parquet", neutral)
    predictions = predict(test, records, popularity)
    submission = test[["molecule_id"]].drop_duplicates().copy()
    submission["smiles"] = submission.molecule_id.map(predictions)
    if submission.smiles.isna().any() or submission.molecule_id.duplicated().any():
        raise ValueError("Incomplete or duplicate predictions")
    if submission.smiles.str.count(";").max() >= 25:
        raise ValueError("More than 25 candidates")
    submission.to_csv(output, index=False)
    print(f"Wrote {output}: {len(submission)} molecules", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    main(args.data_dir, args.output)
