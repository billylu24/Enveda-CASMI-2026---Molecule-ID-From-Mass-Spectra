"""Shared candidate pools, reference retrieval, and ranking metrics."""
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from baseline import ADDUCT_MASS, make_matrix, vectorize
from casmi_ml.data import fingerprint, write_json
from hybrid import blend


def center_mass(group):
    values = [float(r.precursor_mz) - ADDUCT_MASS.get(r.adduct, math.nan) for r in group.itertuples()]
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else None


class CandidateIndex:
    def __init__(self, catalog):
        self.catalog = catalog.sort_values(['mass', 'inchikey14']).reset_index(drop=True)
        self.mass = self.catalog.mass.to_numpy()
        self.cache = {}

    def query(self, center):
        if center is None:
            return self.catalog.iloc[:0]
        delta = max(center * 35e-6, .006)
        return self.catalog.iloc[np.searchsorted(self.mass, center-delta):np.searchsorted(self.mass, center+delta, side='right')]

    def fps(self, candidates):
        ids, fps = [], []
        for idx, row in candidates.iterrows():
            key = row.inchikey14
            if key not in self.cache:
                self.cache[key] = fingerprint(row.normalized_smiles)
            fp = self.cache[key]
            if fp is not None:
                ids.append(idx)
                fps.append(fp)
        return self.catalog.loc[ids], np.asarray(fps, dtype=np.float32).reshape(-1, 2048)


def build_candidates(catalog, coconut_path, final=False):
    library = catalog if final else catalog[catalog.split == 'train']
    library = library[['inchikey14', 'normalized_smiles', 'mass']].copy()
    library['origin'] = 'library'
    coconut = pd.read_parquet(coconut_path, columns=['inchikey', 'canonical_smiles', 'exact_mass'])
    coconut = coconut.rename(columns={'canonical_smiles': 'normalized_smiles', 'exact_mass': 'mass'})
    coconut['inchikey14'] = coconut.inchikey.str[:14]
    coconut['origin'] = 'coconut'
    pool = pd.concat([library, coconut[library.columns]], ignore_index=True)
    pool = pool[pool.inchikey14.notna() & pool.normalized_smiles.notna() & np.isfinite(pool.mass) & (pool.mass > 0)]
    return pool.drop_duplicates('inchikey14').reset_index(drop=True)



def spectrum_signature(mzs, intensities):
    return hashlib.sha256(np.asarray(mzs, dtype=np.float64).tobytes() +
                          np.asarray(intensities, dtype=np.float64).tobytes()).digest()

def build_reference(train_path, catalog, queries, destination, final=False, excluded_source=None, exclude_queries=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    allowed = set(catalog.inchikey14 if final else catalog.loc[catalog.split == 'train', 'inchikey14'])
    centers = np.asarray(sorted({m for _, g in queries.groupby('inchikey14') if (m := center_mass(g)) is not None}))
    if not len(centers):
        raise ValueError('No supported precursor masses')
    mass_by_key = dict(zip(catalog.inchikey14, catalog.mass))
    signatures = {}
    if exclude_queries:
        for key, group in queries.groupby('inchikey14'):
            signatures[key] = {spectrum_signature(r.ms2_mzs, r.ms2_normalized_intensities) for r in group.itertuples()}
    vectors, keys, smiles, masses, chunks = [], [], [], [], []
    columns = ['inchikey14', 'normalized_smiles', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_error_ppm', 'ingest_lib'] if excluded_source else ['inchikey14', 'normalized_smiles', 'ms2_mzs', 'ms2_normalized_intensities', 'precursor_error_ppm']
    for batch_no, batch in enumerate(pq.ParquetFile(train_path).iter_batches(batch_size=8192, columns=columns)):
        for r in batch.to_pandas().itertuples():
            if excluded_source and r.ingest_lib == excluded_source:
                continue
            if r.inchikey14 not in allowed:
                continue
            if r.inchikey14 in signatures and spectrum_signature(r.ms2_mzs, r.ms2_normalized_intensities) in signatures[r.inchikey14]:
                continue
            mass = mass_by_key[r.inchikey14]
            pos = np.searchsorted(centers, mass)
            distance = min(abs(mass - centers[min(pos, len(centers)-1)]), abs(mass - centers[max(0, pos-1)]))
            if distance > max(mass * 35e-6, .006):
                continue
            if pd.notna(r.precursor_error_ppm) and abs(r.precursor_error_ppm) > 30:
                continue
            vector = vectorize(r.ms2_mzs, r.ms2_normalized_intensities)
            if vector:
                vectors.append(vector)
                keys.append(r.inchikey14)
                smiles.append(r.normalized_smiles)
                masses.append(mass)
        if vectors:
            chunks.append(make_matrix(vectors))
            vectors.clear()
        if batch_no % 50 == 0:
            print(f'reference scan {batch_no * 8192:,}; retained {len(keys):,}', flush=True)
    matrix = sparse.vstack(chunks, format="csr") if chunks else make_matrix([])
    sparse.save_npz(destination / 'spectra.npz', matrix)
    pd.DataFrame({'inchikey14': keys, 'normalized_smiles': smiles, 'mass': masses}).to_parquet(destination / 'rows.parquet', index=False)
    write_json(destination / 'complete.json', {'final': final, 'excluded_source': excluded_source, 'exclude_queries': exclude_queries, 'spectra': len(keys)})


class ReferenceIndex:
    def __init__(self, directory):
        directory = Path(directory)
        self.rows = pd.read_parquet(directory / 'rows.parquet')
        self.matrix = sparse.load_npz(directory / 'spectra.npz')
        self.order = np.argsort(self.rows.mass.to_numpy())
        self.mass = self.rows.mass.to_numpy()[self.order]

    def rank(self, group, center, fallback=False):
        if center is None:
            indexes = self.order[:150] if fallback else np.array([], int)
        else:
            delta = max(center * 35e-6, .006)
            indexes = self.order[np.searchsorted(self.mass, center-delta):np.searchsorted(self.mass, center+delta, side='right')]
            if fallback and not len(indexes):
                pos = np.searchsorted(self.mass, center)
                indexes = self.order[max(0, pos-150):pos+150]
        if not len(indexes):
            return []
        vectors = [vectorize(r.ms2_mzs, r.ms2_normalized_intensities) for r in group.itertuples()]
        scores = (make_matrix(vectors) @ self.matrix[indexes].T).toarray()
        columns = defaultdict(list)
        for col, key in enumerate(self.rows.iloc[indexes].inchikey14):
            columns[key].append(col)
        results = []
        for key, cols in columns.items():
            per_query = scores[:, cols].max(1)
            score = .65 * per_query.max() + .35 * np.sort(per_query)[-3:].mean()
            results.append((key, float(score)))
        return sorted(results, key=lambda x: (-x[1], x[0]))


def baseline_rank(group, candidates, fps, reference, center):
    library = reference.rank(group, center)
    candidate_keys = set(candidates.inchikey14)
    library = [(k, s) for k, s in library if k in candidate_keys]
    key_to_fp = {k: fp for k, fp in zip(candidates.inchikey14, fps)}
    analog = []
    refs = [(key_to_fp[k], score) for k, score in library[:25] if k in key_to_fp]
    if refs and len(fps):
        ref_fp = np.asarray([x[0] for x in refs])
        intersection = fps @ ref_fp.T
        union = fps.sum(1)[:, None] + ref_fp.sum(1)[None, :] - intersection
        similarity = intersection / np.maximum(union, 1)
        scores = (similarity * np.asarray([.5 + .5 * x[1] for x in refs])[None, :]).max(1)
        ppm = np.abs(candidates.mass.to_numpy() - center) / center * 1e6
        scores *= np.exp(-.5 * (ppm / 15) ** 2)
        analog = sorted(zip(candidates.inchikey14, scores.tolist()), key=lambda x: (-x[1], x[0]))
    # Reuse the established mixing rule while supplying the common candidate pool.
    ranked = blend([(k, k, s) for k, s in library], [(k, k, s) for k, s in analog])
    output = [k for k, _ in ranked]
    # Keep all remaining candidates for fusion, with deterministic ties.
    output.extend(k for k, _ in analog if k not in set(output))
    seen = set(output)
    output.extend(k for k in candidates.inchikey14 if k not in seen)
    return output


def neural_rank(probability, keys, fps):
    p = np.clip(np.asarray(probability, np.float64), 1e-6, 1 - 1e-6)
    score = fps @ (np.log(p) - np.log1p(-p)) / 2048
    return [keys[i] for i in sorted(range(len(keys)), key=lambda i: (-score[i], keys[i]))]


def rrf(rankings, weights=None):
    if weights is None:
        weights = [1 / len(rankings)] * len(rankings)
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, key in enumerate(ranking, 1):
            scores[key] += weight / (60 + rank)
    return sorted(scores, key=lambda k: (-scores[k], k))


def metrics(rankings, candidates):
    rows = []
    for truth, ranking in rankings.items():
        top = ranking[:25]
        rank = top.index(truth) + 1 if truth in top else 0
        rows.append({'key': truth, 'reciprocal_rank': 1 / rank if rank else 0.,
                     'top1': int(rank == 1), 'top5': int(0 < rank <= 5),
                     'top25': int(rank > 0), 'covered': int(truth in candidates[truth])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('Cannot evaluate an empty split')
    report = {'molecules': len(frame), 'candidate_recall': float(frame.covered.mean()),
              'mrr25': float(frame.reciprocal_rank.mean()),
              **{k: float(frame[k].mean()) for k in ['top1', 'top5', 'top25']},
              'conditional_mrr25': float(frame.loc[frame.covered == 1, 'reciprocal_rank'].mean()) if frame.covered.any() else None}
    return report, frame


class Evaluation:
    def __init__(self, frame, pool, reference):
        self.frame = frame
        self.groups = frame.groupby('inchikey14', sort=True).indices
        self.candidates, self.fps, self.baseline, self.confidence = {}, {}, {}, {}
        index = CandidateIndex(pool)
        for key, ids in self.groups.items():
            group = frame.iloc[ids]
            center = center_mass(group)
            candidates, fps = index.fps(index.query(center))
            self.candidates[key] = candidates.inchikey14.tolist()
            self.fps[key] = fps.astype(np.uint8)
            self.baseline[key] = baseline_rank(group, candidates, fps, reference, center)
            library = reference.rank(group, center)
            self.confidence[key] = library[0][1] if library else 0.

    def rank(self, probabilities):
        return {key: neural_rank(probabilities[ids].mean(0), self.candidates[key], self.fps[key])
                for key, ids in self.groups.items()}
