"""Exact library membership without materializing unneeded reference peak vectors."""

import numpy as np
import pyarrow.parquet as pq

from baseline import formula_mass


def usable_spectrum(mzs, intensities):
    """Same nonempty criterion as baseline.vectorize, with no bins or norm required."""
    masses = np.asarray(mzs, dtype=np.float64)
    values = np.asarray(intensities, dtype=np.float64)
    valid = np.isfinite(masses) & np.isfinite(values) & (values >= 0.01)
    valid &= (masses >= 40) & (masses < 1250)
    return bool(valid.any())


def load_observed_keys(train_path, target_masses, tolerance_ppm=35):
    targets = np.sort(np.asarray(target_masses, dtype=np.float64))
    if not len(targets):
        raise ValueError("Nonempty target masses required")
    columns = [
        "molecular_formula",
        "normalized_smiles",
        "inchikey14",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_error_ppm",
    ]
    formula_cache, observed = {}, set()
    for batch in pq.ParquetFile(train_path).iter_batches(
        batch_size=8192, columns=columns
    ):
        formulas = batch.column(0).to_pylist()
        for formula in set(formulas):
            if formula not in formula_cache:
                formula_cache[formula] = formula_mass(formula)
        masses = np.asarray([formula_cache[f] for f in formulas], dtype=np.float64)
        positions = np.searchsorted(targets, masses)
        left = targets[np.maximum(positions - 1, 0)]
        right = targets[np.minimum(positions, len(targets) - 1)]
        distance = np.minimum(abs(masses - left), abs(masses - right))
        keep = np.flatnonzero(
            np.isfinite(masses) & (distance <= masses * tolerance_ppm * 1e-6)
        )
        if not len(keep):
            continue
        selected = batch.take(keep)
        for _, smiles, key, mzs, intensities, ppm in zip(
            *[selected.column(i).to_pylist() for i in range(6)]
        ):
            if not smiles or not key or not mzs or not intensities:
                continue
            if ppm is not None and abs(ppm) > 30:
                continue
            if usable_spectrum(mzs, intensities):
                observed.add(str(key))
    return observed
