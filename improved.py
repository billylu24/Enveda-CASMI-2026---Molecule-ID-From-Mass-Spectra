"""CASMI 2026 library retrieval with multi-spectrum and accurate-peak reranking."""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from baseline import ADDUCT_MASS, formula_mass, make_matrix, vectorize


def clean_peaks(mzs, intensities, precursor, max_peaks=100):
    mzs = np.asarray(mzs, dtype=np.float32)
    intensities = np.asarray(intensities, dtype=np.float32)
    keep = np.isfinite(mzs) & np.isfinite(intensities)
    keep &= (mzs >= 40) & (mzs <= min(1250, precursor + 2)) & (intensities >= .01)
    mzs, intensities = mzs[keep], intensities[keep]
    if len(mzs) > max_peaks:
        top = np.argpartition(intensities, -max_peaks)[-max_peaks:]
        mzs, intensities = mzs[top], intensities[top]
    weights = np.sqrt(intensities) * np.sqrt(mzs / 100)
    norm = np.linalg.norm(weights)
    if norm:
        weights /= norm
    order = np.argsort(mzs)
    return mzs[order], weights[order]


def scan_library(path, targets, excluded_source=None):
    targets = np.sort(np.asarray(targets, dtype=np.float64))
    parquet = pq.ParquetFile(path)
    columns = ["molecular_formula", "normalized_smiles", "inchikey14",
               "ms2_mzs", "ms2_normalized_intensities", "precursor_error_ppm",
               "precursor_mz", "adduct", "ingest_lib"]
    cache = {}
    records = []
    for batch_no, batch in enumerate(parquet.iter_batches(batch_size=8192, columns=columns)):
        formulas = batch.column(0).to_pylist()
        for formula in set(formulas):
            if formula not in cache:
                cache[formula] = formula_mass(formula)
        masses = np.asarray([cache[f] for f in formulas])
        pos = np.searchsorted(targets, masses)
        distances = np.minimum(abs(masses - targets[np.maximum(pos - 1, 0)]),
                               abs(masses - targets[np.minimum(pos, len(targets) - 1)]))
        indexes = np.flatnonzero(np.isfinite(masses) & (distances <= masses * 35e-6))
        if len(indexes):
            selected = batch.take(indexes)
            for formula, smiles, key, mzs, intensities, ppm, precursor, adduct, source in zip(*[
                selected.column(i).to_pylist() for i in range(len(columns))
            ]):
                if not key or not smiles or not mzs or not intensities or not precursor:
                    continue
                if source == excluded_source or (ppm is not None and abs(ppm) > 30):
                    continue
                clean_mzs, clean_weights = clean_peaks(mzs, intensities, precursor)
                if not len(clean_mzs):
                    continue
                vector = vectorize(mzs, intensities)
                records.append((cache[formula], str(key), str(smiles), vector,
                                clean_mzs, clean_weights, float(precursor),
                                str(adduct), str(source)))
        if batch_no % 50 == 0:
            print(f"scanned {batch_no * 8192:,}; kept {len(records):,}", flush=True)
    print(f"library candidates: {len(records):,}", flush=True)
    return records


def read_np_validation(path):
    parquet = pq.ParquetFile(path)
    columns = ["ingest_lib", "inchikey14", "ms2_mzs",
               "ms2_normalized_intensities", "precursor_mz", "adduct"]
    batches = []
    for batch in parquet.iter_batches(batch_size=10000, columns=columns):
        source = np.asarray(batch.column(0).to_pylist(), dtype=object)
        selected = np.flatnonzero(source == "enveda-np-examples")
        if len(selected):
            batches.append(batch.take(selected).to_pandas())
    data = pd.concat(batches, ignore_index=True)
    data = data.rename(columns={"inchikey14": "molecule_id"})
    print(f"validation: {len(data)} spectra, {data.molecule_id.nunique()} structures", flush=True)
    return data


def matched_dot(a_mz, a_w, b_mz, b_w, tolerance=.025):
    if not len(a_mz) or not len(b_mz):
        return 0.0
    pos = np.searchsorted(b_mz, a_mz)
    lo = np.maximum(pos - 1, 0)
    hi = np.minimum(pos, len(b_mz) - 1)
    dlo = abs(a_mz - b_mz[lo])
    dhi = abs(a_mz - b_mz[hi])
    nearest = np.where(dlo <= dhi, lo, hi)
    delta = np.minimum(dlo, dhi)
    good = delta <= np.maximum(tolerance, a_mz * 15e-6)
    return float(np.sum(a_w[good] * b_w[nearest[good]]))


def pair_score(query, candidate):
    qm, qw, qp, qa = query
    cm, cw, cp, ca = candidate
    fragment = matched_dot(qm, qw, cm, cw)
    qloss = qp - qm
    closs = cp - cm
    qidx = np.argsort(qloss)
    cidx = np.argsort(closs)
    loss = matched_dot(qloss[qidx], qw[qidx], closs[cidx], cw[cidx])
    same_mode = qa.endswith("+") == ca.endswith("+")
    if not same_mode:
        fragment *= .7
    return .7 * fragment + .3 * loss


def rank_group(group, records, matrix, masses, order, mode="enhanced"):
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
    queries = [(clean_peaks(row.ms2_mzs, row.ms2_normalized_intensities,
                           row.precursor_mz), row.precursor_mz, row.adduct)
               for row in group.itertuples()]
    query_vectors = [vectorize(row.ms2_mzs, row.ms2_normalized_intensities)
                     for row in group.itertuples()]
    sims = (make_matrix(query_vectors) @ matrix[indexes].T).toarray()
    by_key = defaultdict(list)
    for col, index in enumerate(indexes):
        by_key[records[index][1]].append(col)
    coarse = {}
    for key, cols in by_key.items():
        per_query = sims[:, cols].max(axis=1)
        peak = float(per_query.max())
        # The repeated observations support a candidate, without suppressing a
        # strong match when the other acquisitions carry little information.
        consensus = .65 * peak + .35 * float(np.mean(np.sort(per_query)[-3:]))
        coarse[key] = (peak if mode == "baseline" else consensus,
                       cols[int(np.argmax(sims[:, cols].max(axis=0)))])
    shortlist = sorted(coarse, key=lambda k: coarse[k][0], reverse=True)[:100]
    scores = {}
    for key in shortlist:
        coarse_score, _ = coarse[key]
        if mode == "baseline" or mode == "consensus":
            scores[key] = coarse_score
            continue
        cols = by_key[key]
        # Score the three best library spectra per structure at high m/z
        # resolution. The full library is still available for the coarse step.
        best_cols = sorted(cols, key=lambda c: float(sims[:, c].max()), reverse=True)[:3]
        detailed = []
        for (qm_qw, qp, qa) in queries:
            qm, qw = qm_qw
            detailed.append(max(pair_score((qm, qw, qp, qa),
                                           (records[indexes[col]][4], records[indexes[col]][5],
                                            records[indexes[col]][6], records[indexes[col]][7]))
                                for col in best_cols))
        exact = .65 * max(detailed) + .35 * float(np.mean(sorted(detailed)[-3:]))
        scores[key] = .5 * coarse_score + .5 * exact
    ordered = sorted(scores, key=scores.get, reverse=True)
    representative = {records[indexes[coarse[key][1]]][1]:
                      records[indexes[coarse[key][1]]][2] for key in ordered}
    return [(key, representative[key]) for key in ordered[:25]]


def predict(data, records, mode="enhanced"):
    if not records:
        raise RuntimeError("No candidates")
    masses = np.asarray([r[0] for r in records])
    order = np.argsort(masses)
    matrix = make_matrix([r[3] for r in records])
    output = {}
    for i, (molecule_id, group) in enumerate(data.groupby("molecule_id", sort=False)):
        output[molecule_id] = rank_group(group, records, matrix, masses, order, mode)
        if i % 100 == 0:
            print(f"ranked {i + 1}/{data.molecule_id.nunique()}", flush=True)
    return output


def mrr(predictions):
    reciprocal = []
    for truth, ranked in predictions.items():
        found = next((1 / (i + 1) for i, (key, _) in enumerate(ranked) if key == truth), 0)
        reciprocal.append(found)
    return float(np.mean(reciprocal)), float(np.mean(np.asarray(reciprocal) == 1))


def main(data_dir, output, validation=False):
    data_dir = Path(data_dir)
    if not (data_dir / "train.parquet").exists():
        matches = list(Path("/kaggle/input").rglob("train.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError("Competition train.parquet not mounted")
        data_dir = matches[0].parent
    if validation:
        data = read_np_validation(data_dir / "train.parquet")
        excluded = "enveda-np-examples"
    else:
        data = pd.read_parquet(data_dir / "test.parquet")
        excluded = None
    target = data.precursor_mz.to_numpy() - data.adduct.map(ADDUCT_MASS).to_numpy()
    target = target[np.isfinite(target)]
    records = scan_library(data_dir / "train.parquet", target, excluded)
    modes = ["baseline", "consensus", "enhanced"] if validation else ["enhanced"]
    for mode in modes:
        ranked = predict(data, records, mode)
        if validation:
            keys = sorted(ranked)
            for split_name, split_keys in [("dev", keys[::2]), ("holdout", keys[1::2])]:
                scores = mrr({key: ranked[key] for key in split_keys})
                print(f"{mode} {split_name}: MRR@25={scores[0]:.4f}, top1={scores[1]:.4f}", flush=True)
        else:
            submission = data[["molecule_id"]].drop_duplicates().copy()
            submission["smiles"] = submission.molecule_id.map(
                lambda key: ";".join(smiles for _, smiles in ranked[key]))
            if submission.smiles.isna().any() or submission.smiles.str.count(";").max() >= 25:
                raise ValueError("Invalid submission")
            submission.to_csv(output, index=False)
            print(f"Wrote {output}: {len(submission)} molecules", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submission_v2.csv")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.output, args.validation)
