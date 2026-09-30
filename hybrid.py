"""CASMI spectrum retrieval plus COCONUT structural analog candidates."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from baseline import ADDUCT_MASS, load_candidates, make_matrix, vectorize


RDLogger.DisableLog("rdApp.*")
FINGERPRINTER = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles, cache):
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles)
        cache[smiles] = FINGERPRINTER.GetFingerprint(mol) if mol else None
    return cache[smiles]


def library_rank(group, records, matrix, masses, order):
    neutral = np.asarray([row.precursor_mz - ADDUCT_MASS.get(row.adduct, math.nan)
                          for row in group.itertuples()])
    neutral = neutral[np.isfinite(neutral)]
    center = float(np.median(neutral)) if len(neutral) else 0.0
    sorted_masses = masses[order]
    delta = max(center * 35e-6, .006)
    start = np.searchsorted(sorted_masses, center - delta)
    stop = np.searchsorted(sorted_masses, center + delta, side="right")
    indexes = order[start:stop]
    if not len(indexes):
        nearest = np.searchsorted(sorted_masses, center)
        indexes = order[max(0, nearest - 150):nearest + 150]
    query_vectors = [vectorize(row.ms2_mzs, row.ms2_normalized_intensities)
                     for row in group.itertuples()]
    scores = (make_matrix(query_vectors) @ matrix[indexes].T).toarray()
    by_key = defaultdict(list)
    for col, index in enumerate(indexes):
        by_key[records[index][1]].append(col)
    ranked = []
    for key, cols in by_key.items():
        per_query = scores[:, cols].max(axis=1)
        score = .65 * float(per_query.max()) + .35 * float(np.mean(np.sort(per_query)[-3:]))
        best_col = cols[int(np.argmax(scores[:, cols].max(axis=0)))]
        ranked.append((key, records[indexes[best_col]][2], score))
    ranked.sort(key=lambda item: (-item[2], item[0]))
    return center, ranked


def coconut_rank(center, library, coconut, sorted_mass, mass_order, fp_cache):
    delta = max(center * 15e-6, .004)
    ids = mass_order[np.searchsorted(sorted_mass, center - delta):
                     np.searchsorted(sorted_mass, center + delta, side="right")]
    if not len(ids):
        return []
    refs = [(fingerprint(smi, fp_cache), score) for _, smi, score in library[:25]]
    refs = [(fp, score) for fp, score in refs if fp is not None]
    if not refs:
        return []
    ref_fps = [fp for fp, _ in refs]
    ranked = {}
    for i in ids:
        row = coconut.iloc[i]
        if not isinstance(row.inchikey, str) or not isinstance(row.canonical_smiles, str):
            continue
        fp = fingerprint(row.canonical_smiles, fp_cache)
        if fp is None:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        score = max(sim * (.5 + .5 * reference_score)
                    for sim, (_, reference_score) in zip(similarities, refs))
        ppm = abs(float(row.exact_mass) - center) / center * 1e6
        score *= math.exp(-.5 * (ppm / 15) ** 2)
        key = row.inchikey[:14]
        if key not in ranked or score > ranked[key][1]:
            ranked[key] = (row.canonical_smiles, score)
    return sorted([(key, smiles, score) for key, (smiles, score) in ranked.items()],
                  key=lambda item: -item[2])


def blend(library, coconut):
    # Holdout validation: protect the strongest library hits, then reserve
    # space for structures that have no observed reference spectrum.
    protect = 2 if library and library[0][2] >= .75 else 1
    output, seen = [], set()
    lib_pos = coco_pos = 0

    def add(source, position):
        while position < len(source) and source[position][0] in seen:
            position += 1
        if position < len(source):
            key, smiles, _ = source[position]
            seen.add(key)
            output.append((key, smiles))
            position += 1
        return position

    for _ in range(protect):
        lib_pos = add(library, lib_pos)
    while len(output) < 25 and (lib_pos < len(library) or coco_pos < len(coconut)):
        old_len = len(output)
        for _ in range(5):
            if len(output) < 25:
                coco_pos = add(coconut, coco_pos)
        if len(output) < 25:
            lib_pos = add(library, lib_pos)
        if len(output) == old_len:
            break
    return output[:25]


def predict(test, records, coconut):
    masses = np.asarray([record[0] for record in records])
    order = np.argsort(masses)
    matrix = make_matrix([record[3] for record in records])
    coconut_masses = coconut.exact_mass.to_numpy()
    coconut_order = np.argsort(coconut_masses)
    sorted_coconut = coconut_masses[coconut_order]
    fp_cache = {}
    output = {}
    for count, (molecule_id, group) in enumerate(test.groupby("molecule_id", sort=False)):
        center, library = library_rank(group, records, matrix, masses, order)
        new_structures = coconut_rank(center, library, coconut, sorted_coconut,
                                      coconut_order, fp_cache)
        output[molecule_id] = ";".join(smiles for _, smiles in blend(library, new_structures))
        if count % 100 == 0:
            print(f"predicted {count + 1}/{test.molecule_id.nunique()}; "
                  f"fingerprint cache {len(fp_cache):,}", flush=True)
    return output


def main(data_dir, coconut_path, output):
    data_dir = Path(data_dir)
    if not (data_dir / "train.parquet").exists():
        matches = list(Path("/kaggle/input").rglob("train.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError("Competition train.parquet not mounted")
        data_dir = matches[0].parent
    coconut_path = Path(coconut_path)
    if not coconut_path.exists():
        matches = list(Path("/kaggle/input").rglob("coconut_structures.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError("COCONUT structure dataset not mounted")
        coconut_path = matches[0]
    test = pd.read_parquet(data_dir / "test.parquet")
    coconut = pd.read_parquet(coconut_path,
                              columns=["canonical_smiles", "exact_mass", "inchikey"])
    neutral = test.precursor_mz.to_numpy() - test.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral)]
    print(f"Test: {len(test):,} spectra, {test.molecule_id.nunique()} molecules", flush=True)
    print(f"COCONUT: {len(coconut):,} structures", flush=True)
    records, _ = load_candidates(data_dir / "train.parquet", neutral)
    predictions = predict(test, records, coconut)
    submission = test[["molecule_id"]].drop_duplicates().copy()
    submission["smiles"] = submission.molecule_id.map(predictions)
    if submission.smiles.isna().any() or (submission.smiles == "").any():
        raise ValueError("Missing predictions")
    if submission.smiles.str.count(";").max() >= 25:
        raise ValueError("More than 25 candidates")
    submission.to_csv(output, index=False)
    print(f"Wrote {output}: {len(submission)} molecules", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--coconut", default="external/coconut_structures.parquet")
    parser.add_argument("--output", default="submission_v3.csv")
    args = parser.parse_args()
    main(args.data_dir, args.coconut, args.output)
