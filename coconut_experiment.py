"""Evaluate natural-product structure retrieval when the true library spectrum is absent."""

import math
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from baseline import ADDUCT_MASS, make_matrix, vectorize
from improved import read_np_validation, scan_library, mrr


RDLogger.DisableLog("rdApp.*")
FINGERPRINTER = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles, cache):
    if smiles not in cache:
        mol = Chem.MolFromSmiles(smiles)
        cache[smiles] = FINGERPRINTER.GetFingerprint(mol) if mol else None
    return cache[smiles]


def library_ranking(group, records, matrix, masses, order, masked_key=None):
    neutral = np.asarray([r.precursor_mz - ADDUCT_MASS.get(r.adduct, math.nan)
                          for r in group.itertuples()])
    neutral = neutral[np.isfinite(neutral)]
    center = float(np.median(neutral)) if len(neutral) else 0.0
    sorted_mass = masses[order]
    delta = max(center * 35e-6, .006)
    idx = order[np.searchsorted(sorted_mass, center - delta):
                np.searchsorted(sorted_mass, center + delta, side="right")]
    if masked_key:
        idx = np.asarray([i for i in idx if records[i][1] != masked_key])
    q = make_matrix([vectorize(r.ms2_mzs, r.ms2_normalized_intensities)
                     for r in group.itertuples()])
    sims = (q @ matrix[idx].T).toarray()
    by_key = defaultdict(list)
    for col, rec_idx in enumerate(idx):
        by_key[records[rec_idx][1]].append(col)
    output = []
    for key, cols in by_key.items():
        sq = sims[:, cols]
        per_query = sq.max(axis=1)
        score = .65 * float(per_query.max()) + .35 * float(np.mean(np.sort(per_query)[-3:]))
        best_col = cols[int(np.argmax(sq.max(axis=0)))]
        output.append((key, records[idx[best_col]][2], score))
    output.sort(key=lambda item: (-item[2], item[0]))
    return center, output


def coconut_ranking(center, library, coconut, sorted_mass, mass_order, fp_cache):
    delta = max(center * 15e-6, .004)
    ids = mass_order[np.searchsorted(sorted_mass, center - delta):
                     np.searchsorted(sorted_mass, center + delta, side="right")]
    if not len(ids):
        return []
    refs = [(fingerprint(smi, fp_cache), score) for _, smi, score in library[:25]]
    refs = [(fp, score) for fp, score in refs if fp is not None]
    if not refs:
        return []
    cands = []
    for i in ids:
        row = coconut.iloc[i]
        fp = fingerprint(row.canonical_smiles, fp_cache)
        if fp is None:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fp, [r[0] for r in refs])
        score = max(sim * (.5 + .5 * ref_score) for sim, (_, ref_score) in zip(similarities, refs))
        ppm = abs(float(row.exact_mass) - center) / center * 1e6
        score *= math.exp(-.5 * (ppm / 15) ** 2)
        cands.append((row.inchikey[:14], row.canonical_smiles, score))
    best = {}
    for key, smi, score in cands:
        if key not in best or score > best[key][1]:
            best[key] = (smi, score)
    return sorted([(k, smi, score) for k, (smi, score) in best.items()],
                  key=lambda item: -item[2])


def main():
    test = read_np_validation("data/train.parquet")
    target = test.precursor_mz.to_numpy() - test.adduct.map(ADDUCT_MASS).to_numpy()
    target = target[np.isfinite(target)]
    records = scan_library("data/train.parquet", target, "enveda-np-examples")
    masses = np.asarray([r[0] for r in records])
    order = np.argsort(masses)
    matrix = make_matrix([r[3] for r in records])
    coconut = pd.read_parquet("external/coconut_structures.parquet")
    coco_order = np.argsort(coconut.exact_mass.to_numpy())
    sorted_coco = coconut.exact_mass.to_numpy()[coco_order]
    fp_cache = {}
    results = defaultdict(dict)
    detailed_results = []
    for count, (truth, group) in enumerate(test.groupby("molecule_id", sort=False)):
        center, known = library_ranking(group, records, matrix, masses, order)
        _, masked = library_ranking(group, records, matrix, masses, order, truth)
        coco = coconut_ranking(center, masked, coconut, sorted_coco, coco_order, fp_cache)
        results["known"][truth] = [(k, s) for k, s, _ in known[:25]]
        results["class2_coconut"][truth] = [(k, s) for k, s, _ in coco[:25]]
        results["class2_library"][truth] = [(k, s) for k, s, _ in masked[:25]]
        detailed_results.append({"truth": truth, "known": known[:50],
                                 "masked": masked[:50], "coconut": coco[:50]})
        if count % 50 == 0:
            print(f"ranked {count + 1}/250; fingerprint cache {len(fp_cache):,}", flush=True)
    for mode, predictions in results.items():
        keys = sorted(predictions)
        for label, split in [("dev", keys[::2]), ("holdout", keys[1::2])]:
            score = mrr({key: predictions[key] for key in split})
            coverage = sum(any(k == key for k, _ in predictions[key]) for key in split) / len(split)
            print(f"{mode} {label}: MRR@25={score[0]:.4f}, top1={score[1]:.4f}, "
                  f"coverage={coverage:.4f}", flush=True)
    with open("coconut_validation_predictions.pkl", "wb") as handle:
        pickle.dump(detailed_results, handle)


if __name__ == "__main__":
    main()
