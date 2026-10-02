"""Molecule-disjoint data preparation and shared spectrum representations."""
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from baseline import formula_mass

RDLogger.DisableLog('rdApp.*')
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
CATEGORIES = ['adduct', 'ionization_mode', 'instrument_type']
COLUMNS = ['inchikey14', 'normalized_smiles', 'ingest_lib', 'molecular_formula',
           'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct',
           'ionization_mode', 'instrument_type', 'collision_energy_ev', 'precursor_error_ppm']


def digest(value):
    return int.from_bytes(hashlib.sha256(('42:' + str(value)).encode()).digest()[:8], 'big')


def split_key(key, diagnostics=()):
    if key in diagnostics:
        return 'diagnostic'
    value = digest(key) % 10000
    return 'train' if value < 8000 else 'dev' if value < 9000 else 'holdout'


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return FP.GetFingerprintAsNumPy(mol) if mol is not None and mol.GetNumAtoms() else None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def prepare(train_path, root, train_limit=20000, eval_limit=2000, per_molecule=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'{root} is already prepared; use a new directory for another split')
    meta = pd.read_parquet(train_path, columns=['inchikey14', 'normalized_smiles',
                                               'ingest_lib', 'molecular_formula'])
    diagnostics = set(meta.loc[meta.ingest_lib == 'enveda-np-examples', 'inchikey14'])
    catalog = meta.drop_duplicates('inchikey14').copy()
    catalog = catalog[catalog.inchikey14.map(lambda x: isinstance(x, str) and len(x) == 14)]
    catalog['split'] = [split_key(k, diagnostics) for k in catalog.inchikey14]
    catalog['mass'] = catalog.molecular_formula.map(formula_mass)
    catalog = catalog[np.isfinite(catalog.mass) & (catalog.mass > 0)].copy()
    catalog['sample_order'] = catalog.inchikey14.map(digest)
    catalog.sort_values('sample_order', inplace=True)
    catalog.to_parquet(root / 'catalog.parquet', index=False)
    selected = {}
    for split in ['train', 'dev', 'holdout', 'diagnostic']:
        limit = train_limit if split == 'train' else eval_limit
        selected[split] = catalog.loc[catalog.split == split, 'inchikey14'].head(limit).tolist()
    key_split = {k: s for s, keys in selected.items() for k in keys}
    # Deterministic reservoir sampling avoids taking only the first instrument/source.
    rng = np.random.default_rng(42)
    reservoirs, counts = defaultdict(list), defaultdict(int)
    row_id = 0
    for batch in pq.ParquetFile(train_path).iter_batches(batch_size=8192, columns=COLUMNS):
        frame = batch.to_pandas()
        frame['row_id'] = np.arange(row_id, row_id + len(frame))
        row_id += len(frame)
        for record in frame[frame.inchikey14.isin(key_split)].to_dict('records'):
            key = record['inchikey14']
            # Diagnostic queries come only from the original diagnostic source.
            if key_split[key] == 'diagnostic' and record['ingest_lib'] != 'enveda-np-examples':
                continue
            counts[key] += 1
            position = counts[key] - 1 if counts[key] <= per_molecule else int(rng.integers(counts[key]))
            if position < per_molecule:
                if len(reservoirs[key]) < per_molecule:
                    reservoirs[key].append(record)
                else:
                    reservoirs[key][position] = record
        if row_id % (8192 * 50) == 0:
            print(f'prepare scanned {row_id:,}', flush=True)
    summary = {}
    for split, keys in selected.items():
        rows, invalid = [], 0
        for key in keys:
            records = reservoirs[key]
            if not records:
                continue
            fp = fingerprint(records[0]['normalized_smiles'])
            if fp is None:
                invalid += 1
                continue
            for row in records:
                row['fingerprint'] = np.packbits(fp).tobytes()
                rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f'No usable molecules in {split}')
        frame.to_parquet(root / f'{split}.parquet', index=False)
        summary[split] = {'molecules': int(frame.inchikey14.nunique()), 'spectra': len(frame),
                          'invalid_structures': invalid}
    training = pd.read_parquet(root / 'train.parquet')
    preprocessing = fit_preprocessing(training)
    write_json(root / 'preprocessing.json', preprocessing)
    write_json(root / 'manifest.json', {'version': 1, 'source': str(Path(train_path).resolve()),
               'source_bytes': Path(train_path).stat().st_size, 'split_seed': 42,
               'train_limit': train_limit, 'eval_limit': eval_limit, 'per_molecule': per_molecule,
               'fingerprint': {'radius': 2, 'bits': 2048}, 'counts': summary})
    print(json.dumps(summary, indent=2), flush=True)


def numeric_metadata(row):
    precursor = row.get('precursor_mz', np.nan)
    precursor = float(precursor) if precursor is not None else np.nan
    ce = row.get('collision_energy_ev')
    ce = np.asarray([] if ce is None else ce, dtype=float)
    ce = ce[np.isfinite(ce)]
    return np.array([precursor if np.isfinite(precursor) else 0.,
                     ce.mean() if len(ce) else 0., ce.min() if len(ce) else 0.,
                     ce.max() if len(ce) else 0., float(not np.isfinite(precursor)),
                     float(not len(ce))], dtype=np.float32)


def category(value):
    return value if isinstance(value, str) and value else '<missing>'


def fit_preprocessing(frame):
    nums = np.stack([numeric_metadata(r) for r in frame.to_dict('records')])
    mean, std = nums[:, :4].mean(0), nums[:, :4].std(0)
    return {'version': 1, 'mean': mean.tolist(), 'std': np.maximum(std, 1.).tolist(),
            'categories': {c: sorted({category(v) for v in frame[c]}) for c in CATEGORIES},
            'bin_width': 1., 'max_mz': 1250., 'max_peaks': 64,
            'mass_frequencies': [0.01, 0.1, 1., 10.]}


def metadata(row, config):
    nums = numeric_metadata(row)
    nums[:4] = (nums[:4] - config['mean']) / config['std']
    pieces = [nums]
    for c in CATEGORIES:
        vocab = config['categories'][c]
        onehot = np.zeros(len(vocab) + 1, np.float32)
        val = category(row.get(c))
        onehot[vocab.index(val) if val in vocab else len(vocab)] = 1.
        pieces.append(onehot)
    return np.concatenate(pieces)


def clean_spectrum(row):
    mz = np.asarray(row['ms2_mzs'], dtype=np.float32)
    intensity = np.asarray(row['ms2_normalized_intensities'], dtype=np.float32)
    if mz.shape != intensity.shape:
        raise ValueError('Peak masses and intensities differ in length')
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz >= 0) & (mz < 1250) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    if len(intensity):
        intensity = intensity / max(float(intensity.max()), 1e-12)
    return mz, np.sqrt(intensity)


def features(row, config):
    mz, weights = clean_spectrum(row)
    hist = np.zeros(1250, np.float32)
    np.maximum.at(hist, mz.astype(int), weights)
    hist /= max(float(np.linalg.norm(hist)), 1e-12)
    precursor = numeric_metadata(row)[0]
    losses = precursor - mz
    valid = (losses >= 0) & (losses < 1250)
    loss_hist = np.zeros(1250, np.float32)
    np.maximum.at(loss_hist, losses[valid].astype(int), weights[valid])
    loss_hist /= max(float(np.linalg.norm(loss_hist)), 1e-12)
    # Stable ordering makes equal-intensity peak selection deterministic.
    ids = np.argsort(-weights, kind='stable')[:config['max_peaks']]
    peak = np.zeros((config['max_peaks'], 19), np.float32)
    mask = np.zeros(config['max_peaks'], bool)
    if len(ids):
        m, w, loss = mz[ids], weights[ids], losses[ids]
        frequencies = np.asarray(config['mass_frequencies'], np.float32)
        angle_m = m[:, None] * frequencies
        angle_l = loss[:, None] * frequencies
        peak[:len(ids)] = np.concatenate([m[:, None] / 1250, w[:, None], loss[:, None] / 1250,
                                         np.sin(angle_m), np.cos(angle_m),
                                         np.sin(angle_l), np.cos(angle_l)], axis=1)
        mask[:len(ids)] = True
    # One zero token for an empty spectrum avoids fully masked attention NaNs.
    if not mask.any():
        mask[0] = True
    return hist, loss_hist, metadata(row, config), peak, mask


def cache_features(frame, preprocessing, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dim = 6 + sum(len(preprocessing['categories'][c]) + 1 for c in CATEGORIES)
    shapes = {'hist': (len(frame), 1250), 'loss': (len(frame), 1250),
              'meta': (len(frame), dim), 'peaks': (len(frame), 64, 19),
              'mask': (len(frame), 64), 'target': (len(frame), 2048)}
    arrays = {name: np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+',
              dtype=bool if name == 'mask' else np.uint8 if name == 'target' else np.float32,
              shape=shape) for name, shape in shapes.items()}
    for i, row in enumerate(frame.to_dict('records')):
        for name, value in zip(['hist', 'loss', 'meta', 'peaks', 'mask'], features(row, preprocessing)):
            arrays[name][i] = value
        arrays['target'][i] = np.unpackbits(np.frombuffer(row['fingerprint'], dtype=np.uint8))
    for value in arrays.values():
        value.flush()
    frame[['inchikey14', 'row_id']].to_parquet(directory / 'rows.parquet', index=False)
    write_json(directory / 'complete.json', {'rows': len(frame), 'preprocessing': preprocessing})
