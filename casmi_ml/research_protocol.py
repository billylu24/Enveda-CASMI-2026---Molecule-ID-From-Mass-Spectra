"""Fresh, molecule-isolated research cohorts and immutable protocol snapshots."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rdkit import rdBase

from casmi_ml.data import COLUMNS, fingerprint, write_json
from casmi_ml.metfrag import digest

ROOT = Path("artifacts/chemistry_20261001")
CATALOG = Path("artifacts/experiment/catalog.parquet")
TRAIN = Path("artifacts/scale_20260929/train60k.parquet")
ENCODER = Path("artifacts/scale_20260929/runs/wide_enhanced_60000/model.pt")


def freeze(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Frozen configuration changed: {path}; use a new root")
    if not path.exists():
        write_json(path, value)
    return value


def prior_queries(artifacts=Path("artifacts"), exclude=ROOT):
    keys, paths = set(), []
    for directory in sorted(Path(artifacts).iterdir()):
        if not directory.is_dir() or directory.resolve() == Path(exclude).resolve():
            continue
        # Top-level prepared cohorts only: reference catalogs are not query sets.
        for path in sorted(directory.glob("*.parquet")):
            if path.name == "catalog.parquet":
                continue
            if "inchikey14" in pq.ParquetFile(path).schema_arrow.names:
                keys.update(pd.read_parquet(path, columns=["inchikey14"]).inchikey14)
                paths.append(str(path))
    # Nested research roots must not disappear from the exclusion audit.
    for manifest in sorted(Path(artifacts).rglob("cohorts.json")):
        if manifest.parent.resolve() == Path(exclude).resolve():
            continue
        value = json.loads(manifest.read_text())
        for part in value.values():
            if isinstance(part, dict) and "keys" in part:
                keys.update(part["keys"])
        paths.append(str(manifest))
    return keys, paths


def choose_keys(catalog, used, count, split, seed):
    eligible = catalog[
        (catalog.split == split) & ~catalog.inchikey14.isin(used)
    ].inchikey14.unique()
    eligible = sorted(
        eligible, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).digest()
    )
    if len(eligible) < count:
        raise ValueError(
            f"Only {len(eligible)} untouched {split} molecules, requested {count}"
        )
    return eligible[:count]


def sample_queries(source, keys, seed=20261001):
    """Reservoir up to nine, then query-count distribution from visible input only."""
    keys, samples, counts = set(keys), {}, {}
    rng = np.random.default_rng(seed)
    offset = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192, columns=COLUMNS):
        frame = batch.to_pandas()
        frame["row_id"] = np.arange(offset, offset + len(frame))
        offset += len(frame)
        for row in frame[frame.inchikey14.isin(keys)].to_dict("records"):
            k = row["inchikey14"]
            counts[k] = counts.get(k, 0) + 1
            bucket = samples.setdefault(k, [])
            pos = counts[k] - 1 if counts[k] <= 9 else int(rng.integers(counts[k]))
            if pos < 9:
                if len(bucket) < 9:
                    bucket.append(row)
                else:
                    bucket[pos] = row
        if offset % 409600 == 0:
            print("query scan", offset, flush=True)
    distribution = np.array([55, 97, 99, 106, 24, 14, 2, 2, 1], dtype=float)
    rows = []
    for key in sorted(keys):
        group = samples.get(key, [])
        if not group:
            raise ValueError(f"Missing query {key}")
        count = int(rng.choice(np.arange(1, 10), p=distribution / distribution.sum()))
        fp = fingerprint(group[0]["normalized_smiles"])
        if fp is None:
            raise ValueError(f"Invalid query label {key}")
        for row in group[:count]:
            row["fingerprint"] = np.packbits(fp).tobytes()
            rows.append(row)
    return pd.DataFrame(rows)


def prepare(root=ROOT, dev_count=2000, holdout_count=4000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "seed": 20261001,
        "dev_molecules": dev_count,
        "holdout_molecules": holdout_count,
        "train": str(TRAIN),
        "encoder_sha256": digest(ENCODER),
        "cohort_design": "unused dev/holdout keys; same 1-9 query spectra in known/unknown; exact query copies excluded",
        "chemical_weights": [0.0, 0.25, 0.5],
        "high_confidence_protection": 0.5,
        "shortlist": 100,
        "gpu_seconds_per_stage": 86400,
        "selection": "unknown MRR improvement, known MRR >= baseline-.001, known Top1 >= baseline-.005",
        "acceptance": "paired unknown MRR CI95 lower>0, known MRR point delta>=-.001, known Top1 delta>=-.005",
        "no_automatic_submission": True,
    }
    freeze(root / "protocol.json", spec)
    if (root / "cohorts.json").exists():
        manifest = json.loads((root / "cohorts.json").read_text())
        for split in ["researchdev", "researchholdout"]:
            if digest(root / f"{split}.parquet") != manifest[split]["sha256"]:
                raise ValueError("Cohort checksum changed")
        return manifest
    used, paths = prior_queries(exclude=root)
    cat = pd.read_parquet(CATALOG)
    dev = choose_keys(cat, used, dev_count, "dev", 20261001)
    holdout = choose_keys(cat, used | set(dev), holdout_count, "holdout", 20261002)
    frame = sample_queries("data/train.parquet", dev + holdout)
    result = {
        "excluded_paths": paths,
        "training_exclusion": "all prior prepared training/queries; original split isolation",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
            "torch": torch.__version__,
        },
    }
    for split, keys in [("researchdev", dev), ("researchholdout", holdout)]:
        part = frame[frame.inchikey14.isin(keys)].reset_index(drop=True)
        part.to_parquet(root / f"{split}.parquet", index=False)
        result[split] = {
            "keys": keys,
            "molecules": len(keys),
            "spectra": len(part),
            "overlap_prior": len(set(keys) & used),
            "sha256": digest(root / f"{split}.parquet"),
        }
    write_json(root / "cohorts.json", result)
    return result
