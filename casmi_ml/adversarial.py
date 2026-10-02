"""Actual conditional fingerprint GAN, with a matched supervised control.

The generator produces Morgan fingerprint distributions, not molecular graphs.
Only a downstream search of an existing structure catalog yields SMILES. No
claim of novel molecule generation or leaderboard improvement is made here.

Preparation groups all spectra by RDKit tautomer-canonical InChIKey14 before a
new, fixed hash split. The original raw keys remain available for excluding
references. Development BCE selects checkpoints; this module never reads the
acceptance spectra while training or choosing an epoch.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from baseline import vectorize
from casmi_ml.data import CATEGORIES, features, fingerprint, fit_preprocessing, write_json


SEED = 20261002
SPLIT_NAMESPACE = "casmi-conditional-fingerprint-gan-v1:20261002:"
BITS = 2048
DEFAULT_CONFIG = {
    "seed": SEED, "width": 256, "noise_dim": 32, "epochs": 12,
    "warmup_epochs": 2, "adversarial_weight": .05, "pos_weight": 4.,
    "lr": .001, "discriminator_lr": .0005, "batch_size": 256,
    "threads": 4, "prediction_samples": 8, "gradient_clip": 1.,
}
_TAUTOMERS = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=500000)
def canonical_structure(smiles):
    """Return the actual canonical graph and official-style identity, or None."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not mol.GetNumAtoms():
        return None
    try:
        canonical = _TAUTOMERS.Canonicalize(mol)
        identity = Chem.MolToInchiKey(canonical)[:14]
        mass = float(Descriptors.ExactMolWt(canonical))
        if len(identity) != 14 or not math.isfinite(mass) or mass <= 0:
            return None
        return (identity, Chem.MolToSmiles(canonical), mass,
                rdMolDescriptors.CalcMolFormula(canonical), Chem.GetFormalCharge(canonical))
    except (RuntimeError, ValueError):
        return None


def split_identity(identity):
    """Stable identity split independent of row, source, and spectrum order."""
    value = int.from_bytes(hashlib.sha256((SPLIT_NAMESPACE + identity).encode()).digest()[:8], "big")
    bucket = value % 10000
    return ("train" if bucket < 8000 else "dev" if bucket < 9000 else "acceptance"), value


def _visible_exclusions(test_path):
    """Use explicit structure labels when present; never infer unlabeled truth."""
    if test_path is None:
        return set(), set(), {"test_path": None, "status": "not_provided"}
    path = Path(test_path)
    schema = pq.ParquetFile(path).schema_arrow.names
    key_columns = [name for name in ["inchikey14", "inchikey", "identity"] if name in schema]
    smiles_columns = [name for name in ["normalized_smiles", "canonical_smiles", "smiles"] if name in schema]
    if not key_columns and not smiles_columns:
        return set(), set(), {"test_path": str(path), "status": "unlabeled_test_identities_not_identifiable"}
    frame = pd.read_parquet(path, columns=key_columns + smiles_columns)
    raw, identities = set(), set()
    for name in key_columns:
        values = {value[:14] for value in frame[name] if isinstance(value, str) and len(value) >= 14}
        if name == "identity":
            identities.update(values)
        else:
            raw.update(values)
    for name in smiles_columns:
        for value in frame[name]:
            structure = canonical_structure(value)
            if structure is not None:
                identities.add(structure[0])
    return raw, identities, {"test_path": str(path), "status": "explicit_labels_used",
                              "raw_keys": len(raw), "canonical_identities": len(identities)}


def spectrum_signature(mzs, intensities):
    """Conservative equality under the frozen retrieval representation."""
    vector = vectorize(mzs, intensities)
    if not vector:
        return None  # An empty vector is not evidence of a copied test spectrum.
    values = np.asarray(sorted(vector.items()), dtype=np.float64)
    return hashlib.sha256(values.tobytes()).digest()


def _visible_signatures(test_path):
    if test_path is None:
        return set(), {"status": "test_not_provided", "unique_nonempty_signatures": 0}
    path = Path(test_path)
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    columns = ["ms2_mzs", "ms2_normalized_intensities"]
    if not set(columns) <= schema:
        return set(), {"status": "test_has_no_spectrum_columns", "unique_nonempty_signatures": 0}
    frame = pd.read_parquet(path, columns=columns)
    signatures = {spectrum_signature(row.ms2_mzs, row.ms2_normalized_intensities)
                  for row in frame.itertuples(index=False)}
    signatures.discard(None)
    return signatures, {"status": "conservative_content_matching_enabled",
                        "test_spectra": len(frame), "unique_nonempty_signatures": len(signatures)}


def prepare(train_path, root, train_limit=60000, eval_limit=1000, per_molecule=2,
            exclude_raw_keys=(), exclude_identities=(), test_path=None, seed=SEED):
    """Stream paired spectra into a frozen, tautomer-identity-disjoint split.

    catalog.parquet preserves every valid raw key and its canonical identity.
    The selected train/dev/acceptance rows retain inchikey14 and add identity,
    canonical_smiles, fingerprint bytes, and row_id. No embeddings are read.
    """
    if any(not isinstance(v, int) or v <= 0 for v in [train_limit, eval_limit, per_molecule]):
        raise ValueError("Data limits must be positive integers")
    train_path, root = Path(train_path), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    requested_raw = sorted(set(exclude_raw_keys))
    requested_identities = sorted(set(exclude_identities))
    request = {"source": str(train_path.resolve()), "source_bytes": train_path.stat().st_size,
               "train_limit": train_limit, "eval_limit": eval_limit, "per_molecule": per_molecule,
               "seed": seed, "exclude_raw_keys": requested_raw,
               "exclude_identities": requested_identities,
               "test_path": str(Path(test_path).resolve()) if test_path is not None else None}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request") != request:
            raise ValueError("Prepared split configuration changed; use a new directory")
        return manifest
    started = time.monotonic()
    parquet = pq.ParquetFile(train_path)
    schema = set(parquet.schema_arrow.names)
    required = {"inchikey14", "normalized_smiles", "ingest_lib", "precursor_mz", "adduct",
                "ionization_mode", "ms2_mzs", "ms2_normalized_intensities"}
    if not required <= schema:
        raise ValueError(f"Training spectra missing columns: {sorted(required - schema)}")
    raw_smiles, diagnostic_raw = {}, set()
    for number, batch in enumerate(parquet.iter_batches(
            batch_size=16384, columns=["inchikey14", "normalized_smiles", "ingest_lib"]), 1):
        frame = batch.to_pandas()
        diagnostic_raw.update(frame.loc[frame.ingest_lib.eq("enveda-np-examples"), "inchikey14"])
        for key, smiles in frame[["inchikey14", "normalized_smiles"]].drop_duplicates("inchikey14").itertuples(index=False, name=None):
            if isinstance(key, str) and len(key) == 14 and key not in raw_smiles:
                raw_smiles[key] = smiles
        if number % 100 == 0:
            print(f"adversarial metadata: {number * 16384:,} rows, {len(raw_smiles):,} raw keys", flush=True)
    visible_raw, visible_identities, visible_report = _visible_exclusions(test_path)
    visible_signatures, signature_report = _visible_signatures(test_path)
    blocked_raw = set(requested_raw) | visible_raw | diagnostic_raw
    blocked_identities = set(requested_identities) | visible_identities
    catalog_rows, invalid = [], 0
    for number, (key, smiles) in enumerate(raw_smiles.items(), 1):
        structure = canonical_structure(smiles)
        if structure is None:
            invalid += 1
            continue
        identity, canonical, mass, formula, charge = structure
        catalog_rows.append({"inchikey14": key, "identity": identity,
                             "normalized_smiles": smiles, "canonical_smiles": canonical,
                             "mass": mass, "molecular_formula": formula, "formal_charge": charge})
        if key in blocked_raw:
            blocked_identities.add(identity)
        if number % 10000 == 0:
            print(f"adversarial canonicalized {number:,}/{len(raw_smiles):,} structures", flush=True)
    if not catalog_rows:
        raise ValueError("No valid molecular structures")
    catalog = pd.DataFrame(catalog_rows)
    split_map = {identity: ("excluded", 0) if identity in blocked_identities else split_identity(identity)
                 for identity in catalog.identity.unique()}
    catalog["split"] = catalog.identity.map(lambda key: split_map[key][0])
    catalog["sample_order"] = catalog.identity.map(lambda key: split_map[key][1])
    catalog = catalog.sort_values(["sample_order", "identity", "inchikey14"]).reset_index(drop=True)
    catalog.to_parquet(root / "catalog.parquet", index=False)
    selected = {}
    for name in ["train", "dev", "acceptance"]:
        limit = train_limit if name == "train" else eval_limit
        selected[name] = catalog.loc[catalog.split.eq(name), "identity"].drop_duplicates().head(limit).tolist()
        if not selected[name]:
            raise ValueError(f"No canonical identities in {name}; increase the input corpus")
    identity_split = {key: name for name, keys in selected.items() for key in keys}
    selected_catalog = catalog.loc[catalog.identity.isin(identity_split)]
    by_raw = selected_catalog.set_index("inchikey14").to_dict("index")
    identity_by_raw = dict(zip(catalog.inchikey14, catalog.identity))
    columns = ["inchikey14", "normalized_smiles", "ingest_lib", "precursor_mz", "adduct",
               "ionization_mode", "ms2_mzs", "ms2_normalized_intensities"]
    columns += [name for name in ["instrument_type", "collision_energy_ev", "precursor_error_ppm", "spectrum_id"]
                if name in schema]
    rng = np.random.default_rng(seed)
    reservoirs, seen = {}, {}
    row_id, quality_excluded = 0, 0
    matched_raw, matched_identities, matched_spectra = set(), set(), 0
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        frame = batch.to_pandas()
        frame["row_id"] = np.arange(row_id, row_id + len(frame))
        row_id += len(frame)
        if visible_signatures:
            # Match before selecting the cohort. A match under any raw alias
            # excludes the entire official identity from every neural split.
            for row in frame[["inchikey14", "ms2_mzs", "ms2_normalized_intensities"]].itertuples(index=False):
                if row.inchikey14 not in identity_by_raw:
                    continue
                if spectrum_signature(row.ms2_mzs, row.ms2_normalized_intensities) in visible_signatures:
                    matched_raw.add(row.inchikey14)
                    matched_identities.add(identity_by_raw[row.inchikey14])
                    matched_spectra += 1
        frame = frame.loc[frame.inchikey14.isin(by_raw)]
        for record in frame.to_dict("records"):
            metadata = by_raw[record["inchikey14"]]
            ppm = record.get("precursor_error_ppm")
            if ppm is not None and pd.notna(ppm) and abs(float(ppm)) > 30:
                quality_excluded += 1
                continue
            identity = metadata["identity"]
            record["identity"] = identity
            record["canonical_smiles"] = metadata["canonical_smiles"]
            for name in CATEGORIES:
                record.setdefault(name, None)
            record.setdefault("collision_energy_ev", None)
            seen[identity] = seen.get(identity, 0) + 1
            count = seen[identity]
            position = count - 1 if count <= per_molecule else int(rng.integers(count))
            if position < per_molecule:
                reservoir = reservoirs.setdefault(identity, [])
                if position == len(reservoir):
                    reservoir.append(record)
                else:
                    reservoir[position] = record
        if row_id % (8192 * 100) == 0:
            print(f"adversarial spectra: {row_id:,} scanned, {len(reservoirs):,} identities", flush=True)
    if matched_identities:
        blocked_identities.update(matched_identities)
        catalog.loc[catalog.identity.isin(matched_identities), "split"] = "excluded"
        catalog.to_parquet(root / "catalog.parquet", index=False)
        for identity in matched_identities:
            reservoirs.pop(identity, None)
        selected = {name: [key for key in keys if key not in matched_identities]
                    for name, keys in selected.items()}
    signature_report.update({"matched_reference_spectra": matched_spectra,
                             "matched_rawkeys": sorted(matched_raw),
                             "matched_official_identities": sorted(matched_identities),
                             "all_matching_official_identities_excluded_from_neural_fitting": True,
                             "definition": "Identical nonempty normalized retrieval vectors; conservative, may exclude unrelated aliases; not a proof of all unlabeled test identities."})
    counts = {}
    for name, keys in selected.items():
        rows = []
        for identity in keys:
            records = reservoirs.get(identity, [])
            if not records:
                continue
            fp = fingerprint(records[0]["canonical_smiles"])
            if fp is None:
                continue
            packed = np.packbits(fp).tobytes()
            for record in records:
                rows.append({**record, "fingerprint": packed})
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"No usable spectra in {name}")
        frame.to_parquet(root / f"{name}.parquet", index=False)
        counts[name] = {"molecules": int(frame.identity.nunique()),
                        "rawkeys": int(frame.inchikey14.nunique()), "spectra": len(frame)}
    training = pd.read_parquet(root / "train.parquet")
    preprocessing = fit_preprocessing(training)
    write_json(root / "preprocessing.json", preprocessing)
    excluded_raw = sorted(catalog.loc[catalog.split.eq("excluded"), "inchikey14"])
    manifest = {"version": 1, "request": request, "split_namespace": SPLIT_NAMESPACE,
                "identity": "RDKit default TautomerEnumerator.Canonicalize then InChIKey14",
                "rdkit_version": rdBase.rdkitVersion, "counts": counts,
                "catalog_rawkeys": len(catalog), "catalog_identities": int(catalog.identity.nunique()),
                "invalid_structures": invalid, "quality_excluded_spectra": quality_excluded,
                "excluded_rawkeys": excluded_raw, "excluded_identities": sorted(blocked_identities),
                "diagnostic_rawkeys": len(diagnostic_raw), "visible_test_exclusion": visible_report,
                "visible_spectrum_exclusion": signature_report,
                "explicit_exclude_rawkeys_not_found": sorted(set(requested_raw) - set(catalog.inchikey14)),
                "fingerprint": {"radius": 2, "bits": BITS, "graph": "tautomer canonical"},
                "selection": "hash priority by canonical identity; seeded per-identity reservoir",
                "previous_experiment_overlap": "not established; a new split does not itself prove unseen prior auditing",
                "seconds": time.monotonic() - started}
    write_json(manifest_path, manifest)
    return manifest


def condition_features(frame, preprocessing):
    """Deterministic fixed histogram, neutral-loss, and train-fitted metadata."""
    width = 2500 + 6 + sum(len(preprocessing["categories"][name]) + 1 for name in CATEGORIES)
    conditions = np.empty((len(frame), width), dtype=np.float32)
    for index, record in enumerate(frame.to_dict("records")):
        hist, loss, meta, _, _ = features(record, preprocessing)
        conditions[index] = np.concatenate([hist, loss, meta])
    if not np.isfinite(conditions).all():
        raise ValueError("Non-finite spectrum features")
    return conditions


class ConditionalGenerator(nn.Module):
    def __init__(self, condition_dim, width=256, noise_dim=32, bits=BITS):
        super().__init__()
        self.condition_dim, self.width, self.noise_dim, self.bits = condition_dim, width, noise_dim, bits
        self.encoder = nn.Sequential(nn.Linear(condition_dim, width), nn.LayerNorm(width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(width + noise_dim, width), nn.GELU(), nn.Linear(width, bits))

    def forward(self, condition, noise):
        if condition.ndim != 2 or noise.shape != (len(condition), self.noise_dim):
            raise ValueError("Generator condition/noise shapes do not match")
        return self.decoder(torch.cat([self.encoder(condition), noise], dim=1))


class ConditionalDiscriminator(nn.Module):
    """Conditional real/fake *pair* discriminator, with spectral norm bounds."""
    def __init__(self, condition_dim, width=256, bits=BITS):
        super().__init__()
        spectral = nn.utils.parametrizations.spectral_norm
        self.condition = nn.Sequential(spectral(nn.Linear(condition_dim, width)), nn.LeakyReLU(.2))
        self.structure = nn.Sequential(spectral(nn.Linear(bits, width)), nn.LeakyReLU(.2))
        self.pair = nn.Sequential(spectral(nn.Linear(width * 2, width)), nn.LeakyReLU(.2),
                                  spectral(nn.Linear(width, 1)))

    def forward(self, condition, fingerprints):
        return self.pair(torch.cat([self.condition(condition), self.structure(fingerprints)], 1)).squeeze(1)


def straight_through_bernoulli(probability):
    """Hard bits in the forward pass; a biased straight-through G gradient.

    Hard fake bits prevent the discriminator winning merely because real
    fingerprints are binary and generated probabilities are continuous.
    """
    hard = (torch.rand_like(probability) < probability).to(probability.dtype)
    return hard + (probability - probability.detach())


def _check_step(loss, module, optimizer, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite GAN loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def alternating_step(generator, discriminator, g_optimizer, d_optimizer,
                     condition, target, identities, *, adversarial_weight=.05,
                     pos_weight=4., gradient_clip=1., generator_noise=None):
    """One D update followed by one G update; no D update at weight zero."""
    if not 0 <= adversarial_weight <= 1 or not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("Invalid adversarial/positive weight")
    if target.shape != (len(condition), generator.bits):
        raise ValueError("Fingerprint target shape does not match generator")
    if not torch.isfinite(target).all() or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Real fingerprint targets must be finite binary bits")
    if generator_noise is not None:
        if (generator_noise.shape != (len(condition), generator.noise_dim)
                or not torch.isfinite(generator_noise).all()):
            raise ValueError("Invalid fixed generator noise")
        generator_noise = generator_noise.to(device=condition.device, dtype=condition.dtype)
    stats = {"d_loss": None, "d_real_accuracy": None, "d_fake_accuracy": None,
             "d_mismatch_accuracy": None, "g_adversarial_loss": 0., "d_updates": 0, "g_updates": 1}
    if adversarial_weight:
        discriminator.train()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
        d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            probability = torch.sigmoid(generator(condition, torch.randn(len(condition), generator.noise_dim,
                                                                           device=condition.device)))
            fake = (torch.rand_like(probability) < probability).to(probability.dtype)
        real_logits, fake_logits = discriminator(condition, target), discriminator(condition, fake)
        d_loss = .5 * (F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, .9))
                       + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
        # Wrong real structures encourage use of the spectrum condition. A
        # same-identity pair is never assigned a false mismatch label.
        permutation = torch.roll(torch.arange(len(target), device=target.device), 1)
        mismatch = identities != identities[permutation]
        if mismatch.any():
            mismatch_logits = discriminator(condition[mismatch], target[permutation][mismatch])
            d_loss = d_loss + .5 * F.binary_cross_entropy_with_logits(mismatch_logits, torch.zeros_like(mismatch_logits))
            stats["d_mismatch_accuracy"] = float((mismatch_logits.detach() < 0).float().mean())
        _check_step(d_loss, discriminator, d_optimizer, gradient_clip)
        stats.update(d_loss=float(d_loss.detach()), d_real_accuracy=float((real_logits.detach() > 0).float().mean()),
                     d_fake_accuracy=float((fake_logits.detach() < 0).float().mean()), d_updates=1)
        # Freeze D parameters and spectral-normalization updates during G's step.
        discriminator.eval()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
    g_optimizer.zero_grad(set_to_none=True)
    noise = generator_noise if generator_noise is not None else torch.randn(
        len(condition), generator.noise_dim, device=condition.device)
    logits = generator(condition, noise)
    supervised = F.binary_cross_entropy_with_logits(logits, target,
                                                    pos_weight=logits.new_tensor(pos_weight))
    g_adversarial = logits.new_zeros(())
    if adversarial_weight:
        fake = straight_through_bernoulli(torch.sigmoid(logits))
        g_adversarial = F.binary_cross_entropy_with_logits(discriminator(condition, fake),
                                                          torch.ones(len(target), device=target.device))
    g_loss = supervised + adversarial_weight * g_adversarial
    _check_step(g_loss, generator, g_optimizer, gradient_clip)
    if adversarial_weight:
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
    stats.update(g_loss=float(g_loss.detach()), supervised_bce=float(supervised.detach()),
                 g_adversarial_loss=float(g_adversarial.detach()))
    return stats


def _noise_for_conditions(conditions, samples, noise_dim, seed):
    noise = np.empty((len(conditions), samples, noise_dim), dtype=np.float32)
    prefix = f"casmi-gan-noise-v1:{seed}:".encode()
    for index, condition in enumerate(conditions):
        content = hashlib.sha256(prefix + np.ascontiguousarray(condition, dtype=np.float32).tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(content[:8], "big"))
        noise[index] = rng.standard_normal((samples, noise_dim)).astype(np.float32)
    return noise


@torch.inference_mode()
def _predict_conditions(model, conditions, *, samples=8, seed=SEED, batch_size=128, diagnostics=False):
    if not isinstance(samples, int) or not 1 <= samples <= 64 or batch_size <= 0:
        raise ValueError("Prediction sample count must be 1..64 and batch size positive")
    model.eval()
    device = next(model.parameters()).device
    predictions, variances, hamming = [], [], []
    for start in range(0, len(conditions), batch_size):
        batch = np.asarray(conditions[start:start + batch_size], dtype=np.float32)
        noise = _noise_for_conditions(batch, samples, model.noise_dim, seed)
        c = torch.as_tensor(np.repeat(batch, samples, axis=0), device=device)
        z = torch.as_tensor(noise.reshape(-1, model.noise_dim), device=device)
        probabilities = torch.sigmoid(model(c, z)).reshape(len(batch), samples, model.bits)
        predictions.append(probabilities.mean(1).cpu().numpy())
        if diagnostics:
            variances.append(probabilities.var(1, unbiased=False).mean(1).cpu().numpy())
            binary = probabilities >= .5
            hamming.append((binary[:, 1:] != binary[:, :1]).float().mean((1, 2)).cpu().numpy()
                           if samples > 1 else np.zeros(len(batch)))
    output = np.concatenate(predictions) if predictions else np.empty((0, model.bits), np.float32)
    if diagnostics:
        return output, {"noise_probability_variance": float(np.concatenate(variances).mean()) if variances else 0.,
                        "noise_threshold_hamming_fraction": float(np.concatenate(hamming).mean()) if hamming else 0.,
                        "mean_probability_density": float(output.mean()) if len(output) else 0.,
                        "threshold_bit_density": float((output >= .5).mean()) if len(output) else 0.}
    return output


def predict_fingerprints(model, frame, preprocessing, samples=8, seed=SEED, batch_size=128):
    """Return n_spectra x 2048 mean probabilities, with order-stable noise.

    The noise for each spectrum is hashed from its deterministic condition
    features, so batch size, row order, and group iteration cannot alter it.
    """
    conditions = condition_features(frame, preprocessing)
    if conditions.shape[1] != model.condition_dim:
        raise ValueError("Prediction preprocessing does not match checkpoint dimensions")
    return _predict_conditions(model, conditions, samples=samples, seed=seed, batch_size=batch_size)


class _CachedDataset(Dataset):
    def __init__(self, directory):
        self.condition = np.load(Path(directory) / "condition.npy", mmap_mode="r")
        self.target = np.load(Path(directory) / "target.npy", mmap_mode="r")
        self.identity = np.load(Path(directory) / "identity.npy", mmap_mode="r")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (torch.from_numpy(np.array(self.condition[index], copy=True)),
                torch.from_numpy(np.array(self.target[index], dtype=np.float32, copy=True)),
                torch.tensor(int(self.identity[index]), dtype=torch.int64))


def _cache_split(root, split, preprocessing):
    directory = root / "adversarial_features" / split
    marker = directory / "complete.json"
    source = root / f"{split}.parquet"
    stamp = {"source_bytes": source.stat().st_size, "preprocessing": preprocessing}
    if marker.exists() and json.loads(marker.read_text()) == stamp:
        return _CachedDataset(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(source)
    count = parquet.metadata.num_rows
    dim = 2500 + 6 + sum(len(preprocessing["categories"][name]) + 1 for name in CATEGORIES)
    condition = np.lib.format.open_memmap(directory / "condition.npy", mode="w+", dtype=np.float32, shape=(count, dim))
    target = np.lib.format.open_memmap(directory / "target.npy", mode="w+", dtype=np.uint8, shape=(count, BITS))
    identities = np.lib.format.open_memmap(directory / "identity.npy", mode="w+", dtype=np.int64, shape=(count,))
    identity_numbers, start = {}, 0
    columns = ["identity", "fingerprint", "precursor_mz", "adduct", "ionization_mode", "instrument_type",
               "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        frame = batch.to_pandas()
        stop = start + len(frame)
        condition[start:stop] = condition_features(frame, preprocessing)
        for offset, row in enumerate(frame.itertuples(index=False)):
            packed = np.frombuffer(row.fingerprint, dtype=np.uint8)
            if len(packed) != BITS // 8:
                raise ValueError("Prepared fingerprint has wrong bit count")
            target[start + offset] = np.unpackbits(packed)
            if row.identity not in identity_numbers:
                identity_numbers[row.identity] = len(identity_numbers)
            identities[start + offset] = identity_numbers[row.identity]
        start = stop
    for array in [condition, target, identities]:
        array.flush()
    write_json(marker, stamp)
    return _CachedDataset(directory)


def _configure(seed, threads):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    torch.set_num_threads(min(threads, available))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _bce_report(probabilities, targets, positive_weight):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    target = np.asarray(targets, dtype=np.float64)
    loss_positive = -target * np.log(p)
    loss_negative = -(1-target) * np.log1p(-p)
    return {"bce": float((positive_weight * loss_positive + loss_negative).mean()),
            "unweighted_bce": float((loss_positive + loss_negative).mean()),
            "target_bit_density": float(target.mean())}


def load_generator(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "casmi_conditional_fingerprint_gan_v1":
        raise ValueError("Unsupported adversarial checkpoint format")
    model = ConditionalGenerator(checkpoint["condition_dim"], checkpoint["config"]["width"],
                                 checkpoint["config"]["noise_dim"], checkpoint["fingerprint_bits"])
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def train_pair(root, output, config=None, seconds_per_arm=1800., device=None):
    """Train identical supervised and conditional-GAN arms; read train/dev only.

    No acceptance ranking or acceptance fingerprint loss is used here. Run the
    separate release evaluation once after both selected checkpoints are frozen.
    """
    root, output = Path(root), Path(output)
    config = {**DEFAULT_CONFIG, **(config or {})}
    if (not math.isfinite(seconds_per_arm) or seconds_per_arm <= 0 or config["epochs"] <= 0
            or not 0 <= config["warmup_epochs"] < config["epochs"]
            or not 0 < config["adversarial_weight"] <= 1 or config["pos_weight"] <= 0
            or config["width"] <= 0 or config["noise_dim"] <= 0 or config["batch_size"] <= 0):
        raise ValueError("Invalid adversarial training configuration")
    manifest = json.loads((root / "manifest.json").read_text())
    preprocessing = json.loads((root / "preprocessing.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    comparison_path = output / "comparison.json"
    if comparison_path.exists():
        saved = json.loads(comparison_path.read_text())
        if saved["config"] != config or saved["data_manifest"] != manifest:
            raise ValueError("Completed experiment configuration changed; use a new output directory")
        return saved
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _configure(config["seed"], config["threads"])
    train_data = _cache_split(root, "train", preprocessing)
    dev_data = _cache_split(root, "dev", preprocessing)
    probe_ids = np.linspace(0, len(train_data)-1, min(len(train_data), 2048), dtype=int)
    probe_conditions = np.array(train_data.condition[probe_ids], copy=True)
    probe_targets = np.array(train_data.target[probe_ids], copy=True)
    dev_conditions = np.array(dev_data.condition, copy=True)
    dev_targets = np.array(dev_data.target, copy=True)
    arms = {}
    for name, weight in [("baseline", 0.), ("cgan", config["adversarial_weight"])]:
        _configure(config["seed"], config["threads"])
        arm_output = output / name
        arm_output.mkdir(parents=True, exist_ok=True)
        generator = ConditionalGenerator(train_data.condition.shape[1], config["width"], config["noise_dim"]).to(device)
        discriminator = ConditionalDiscriminator(train_data.condition.shape[1], config["width"]).to(device)
        g_optimizer = torch.optim.AdamW(generator.parameters(), lr=config["lr"], weight_decay=1e-4)
        d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config["discriminator_lr"],
                                        betas=(.5, .999), weight_decay=1e-4)
        loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config["seed"]))
        best, history, d_steps, g_steps, best_epoch, best_d_steps = math.inf, [], 0, 0, 0, 0
        started = time.monotonic()
        stop_reason = "epoch_limit"
        for epoch in range(1, config["epochs"] + 1):
            generator.train()
            active_weight = weight if epoch > config["warmup_epochs"] else 0.
            sums, denominators, count, completed = {}, {}, 0, True
            for batch_number, (condition, target, identities) in enumerate(loader):
                if time.monotonic() - started >= seconds_per_arm:
                    completed, stop_reason = False, "time_budget"
                    break
                condition, target, identities = condition.to(device), target.to(device), identities.to(device)
                # Match G's noise across both arms even though D consumes a
                # different number of random values on its separate stream.
                noise_seed = int.from_bytes(hashlib.sha256(
                    f"casmi-gan-training-noise-v1:{config['seed']}:{epoch}:{batch_number}".encode()).digest()[:8], "big")
                noise_rng = torch.Generator(device=device).manual_seed(noise_seed)
                generator_noise = torch.randn(len(condition), generator.noise_dim,
                                              generator=noise_rng, device=device)
                stats = alternating_step(generator, discriminator, g_optimizer, d_optimizer,
                                         condition, target, identities, adversarial_weight=active_weight,
                                         pos_weight=config["pos_weight"], gradient_clip=config["gradient_clip"],
                                         generator_noise=generator_noise)
                size = len(target)
                count += size
                d_steps += stats["d_updates"]
                g_steps += stats["g_updates"]
                for key, value in stats.items():
                    if value is not None and key not in {"d_updates", "g_updates"}:
                        sums[key] = sums.get(key, 0.) + value * size
                        denominators[key] = denominators.get(key, 0) + size
            if not count:
                break
            dev_probabilities, diversity = _predict_conditions(generator, dev_conditions,
                samples=config["prediction_samples"], seed=config["seed"], diagnostics=True)
            train_probabilities = _predict_conditions(generator, probe_conditions,
                samples=config["prediction_samples"], seed=config["seed"])
            dev_report = _bce_report(dev_probabilities, dev_targets, config["pos_weight"])
            train_report = _bce_report(train_probabilities, probe_targets, config["pos_weight"])
            report = {"epoch": epoch, "partial_epoch": not completed, "spectra_seen": count,
                      "active_adversarial_weight": active_weight,
                      "dev_bce": dev_report["bce"], "dev_unweighted_bce": dev_report["unweighted_bce"],
                      "train_probe_bce": train_report["bce"], "train_probe_unweighted_bce": train_report["unweighted_bce"],
                      "bce_gap": dev_report["bce"] - train_report["bce"],
                      "dev_target_bit_density": dev_report["target_bit_density"],
                      **diversity, "d_steps_total": d_steps, "g_steps_total": g_steps,
                      **{key: value/denominators[key] for key, value in sums.items()},
                      "seconds": time.monotonic() - started}
            history.append(report)
            # GAN checkpoint selection only considers epochs that actually used
            # adversarial updates. A warmup-only artifact is not called a GAN.
            eligible = name == "baseline" or (active_weight > 0 and d_steps > 0)
            if eligible and report["dev_bce"] < best:
                best, best_epoch, best_d_steps = report["dev_bce"], epoch, d_steps
                checkpoint = {"format": "casmi_conditional_fingerprint_gan_v1", "arm": name,
                              "condition_dim": generator.condition_dim, "fingerprint_bits": BITS,
                              "config": config, "preprocessing": preprocessing, "epoch": epoch,
                              "d_steps": d_steps, "g_steps": g_steps,
                              "generator_state_dict": generator.state_dict(),
                              "discriminator_state_dict": discriminator.state_dict() if weight else None,
                              "data_manifest": manifest,
                              "checkpoint_selection": "minimum development supervised weighted BCE; acceptance not read",
                              "training_generator_noise": "stateless seed+epoch+batch stream shared by both arms, independent of D RNG",
                              "generation_scope": "conditional fingerprint distributions; requires existing structure retrieval"}
                torch.save(checkpoint, arm_output / "model.pt.tmp")
                (arm_output / "model.pt.tmp").replace(arm_output / "model.pt")
                np.save(arm_output / "dev_probabilities.npy", dev_probabilities)
            write_json(arm_output / "history.json", history)
            print("adversarial", name, json.dumps(report), flush=True)
            if not completed:
                break
        if not best_epoch:
            if name == "cgan":
                raise RuntimeError("Budget ended before an actual adversarial epoch; no GAN checkpoint is claimed")
            raise RuntimeError("Budget ended before any supervised training update")
        arms[name] = {"checkpoint": str(arm_output / "model.pt"), "selected_epoch": best_epoch,
                      "selected_dev_bce": best, "selected_d_steps": best_d_steps,
                      "d_steps_total": d_steps, "g_steps_total": g_steps,
                      "seconds": time.monotonic() - started, "stop_reason": stop_reason,
                      "train_spectra": len(train_data), "dev_spectra": len(dev_data),
                      "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
                      "history": history}
        write_json(arm_output / "result.json", arms[name])
    comparison = {"status": "actual_training_completed_retrieval_validation_pending",
                  "config": config, "device": device, "data_manifest": manifest, "arms": arms,
                  "acceptance_used": False, "identical_generator_architecture": True,
                  "matched_generator_training_noise": True,
                  "baseline_adversarial_weight": 0., "cgan_adversarial_weight": config["adversarial_weight"],
                  "note": "Fingerprint BCE and GAN losses do not establish molecule ranking or leaderboard improvement."}
    write_json(comparison_path, comparison)
    return comparison


def soft_tanimoto_scores(probabilities, candidate_fingerprints):
    """Fixed fingerprint retrieval score, with finite and empty-pool checks."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 2:
        probabilities = probabilities.mean(0)
    fingerprints = np.asarray(candidate_fingerprints, dtype=np.float64)
    if probabilities.shape != (BITS,) or fingerprints.ndim != 2 or fingerprints.shape[1] != BITS:
        raise ValueError("Retrieval requires 2048-bit probabilities and fingerprint rows")
    if not np.isfinite(probabilities).all() or not np.isfinite(fingerprints).all():
        raise ValueError("Non-finite retrieval features")
    if ((probabilities < 0) | (probabilities > 1)).any() or not np.all((fingerprints == 0) | (fingerprints == 1)):
        raise ValueError("Retrieval probabilities must be bounded and candidates binary")
    intersection = fingerprints @ probabilities
    union = fingerprints.sum(1) + probabilities.sum() - intersection
    return intersection / np.maximum(union, 1e-12)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparation = subparsers.add_parser("prepare")
    preparation.add_argument("--train", required=True)
    preparation.add_argument("--root", required=True)
    preparation.add_argument("--test")
    preparation.add_argument("--exclude-keys-json")
    preparation.add_argument("--train-limit", type=int, default=60000)
    preparation.add_argument("--eval-limit", type=int, default=1000)
    preparation.add_argument("--per-molecule", type=int, default=2)
    training = subparsers.add_parser("train")
    training.add_argument("--root", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--seconds-per-arm", type=float, default=1800)
    training.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    if args.command == "prepare":
        exclusions = json.loads(Path(args.exclude_keys_json).read_text()) if args.exclude_keys_json else []
        report = prepare(args.train, args.root, args.train_limit, args.eval_limit, args.per_molecule,
                         exclude_raw_keys=exclusions, test_path=args.test)
    else:
        report = train_pair(args.root, args.output, seconds_per_arm=args.seconds_per_arm, device=args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
