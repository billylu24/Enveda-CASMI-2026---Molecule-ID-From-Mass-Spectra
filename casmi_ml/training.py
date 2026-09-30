"""CPU training, bounded experiments, checkpoints, and inference."""
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from casmi_ml.data import cache_features, write_json
from casmi_ml.models import FingerprintModel, SpectrumDataset
from casmi_ml.ranking import (
    Evaluation,
    ReferenceIndex,
    build_candidates,
    build_reference,
    metrics,
)


def configure(seed=42, threads=12):
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    torch.set_num_threads(min(threads, available))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cached(root, split, preprocessing):
    root = Path(root)
    directory = root / 'features' / split
    marker = directory / 'complete.json'
    if not marker.exists() or json.loads(marker.read_text())['preprocessing'] != preprocessing:
        cache_features(pd.read_parquet(root / f'{split}.parquet'), preprocessing, directory)
    return directory


def load_evaluation(root, split, coconut):
    root = Path(root)
    frame = pd.read_parquet(root / f'{split}.parquet')
    catalog = pd.read_parquet(root / 'catalog.parquet')
    ref_dir = root / 'reference' / split
    if not (ref_dir / 'complete.json').exists():
        manifest = json.loads((root / 'manifest.json').read_text())
        build_reference(manifest['source'], catalog, frame, ref_dir)
    pool = build_candidates(catalog, coconut)
    reference = ReferenceIndex(ref_dir)
    evaluation = Evaluation(frame, pool, reference)
    return evaluation


@torch.inference_mode()
def probabilities(model, dataset, batch_size=128, workers=2):
    model.eval()
    output = []
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        output.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(output)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = FingerprintModel(checkpoint['architecture'], checkpoint['metadata_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


def benchmark(root, seconds=30, threads=12):
    configure(threads=threads)
    root = Path(root)
    prep = json.loads((root / 'preprocessing.json').read_text())
    dataset = SpectrumDataset(cached(root, 'train', prep))
    metadata_dim = dataset.arrays['meta'].shape[1]
    results = {}
    for architecture in ['mlp', 'enhanced', 'metadata', 'deepsets', 'transformer']:
        batch_size = 128 if architecture in ['mlp', 'enhanced', 'metadata'] else 32
        model = FingerprintModel(architecture, metadata_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        start, seen = time.monotonic(), 0
        for batches, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch['target'].float())
            loss.backward()
            optimizer.step()
            seen += len(batch['target'])
            if batches >= 3 and time.monotonic() - start >= seconds:
                break
        elapsed = time.monotonic() - start
        results[architecture] = {'spectra_per_second': seen / elapsed,
                                 'epoch_seconds': len(dataset) * elapsed / seen,
                                 'parameters': sum(p.numel() for p in model.parameters())}
        print('benchmark', architecture, results[architecture], flush=True)
    slowest = max(r['epoch_seconds'] for r in results.values())
    fraction = min(1., 1800 / slowest)
    results['common_train_fraction'] = fraction
    write_json(root / 'benchmark.json', results)
    return results


def train(root, config, output, coconut, evaluation=None, seconds=7200,
          final=False, train_fraction=1., threads=12):
    start = time.monotonic()
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        saved = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError(f'Configuration changed for completed run {output}; use a new root')
        return json.loads((output / 'result.json').read_text())
    seed, architecture = config.get('seed', 42), config['architecture']
    configure(seed, threads)
    prep = json.loads((root / 'preprocessing.json').read_text())
    split = 'final_train' if final else 'train'
    directory = cached(root, split, prep)
    row_info = pd.read_parquet(directory / 'rows.parquet')
    if train_fraction < 1:
        # Same prefix of molecule manifest for every architecture and seed.
        keys = row_info.inchikey14.drop_duplicates().tolist()
        keys = set(keys[:max(1, int(len(keys) * train_fraction))])
        indexes = np.flatnonzero(row_info.inchikey14.isin(keys))
    else:
        indexes = None
    dataset = SpectrumDataset(directory, indexes)
    metadata_dim = dataset.arrays['meta'].shape[1]
    model = FingerprintModel(architecture, metadata_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', .001), weight_decay=1e-4)
    batch_size = 128 if architecture in ['mlp', 'enhanced', 'metadata'] else 32
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                        generator=generator, persistent_workers=True)
    if not final:
        evaluation = evaluation or load_evaluation(root, 'dev', coconut)
        dev_dataset = SpectrumDataset(cached(root, 'dev', prep))
    best, stale, history, best_epoch = -1., 0, [], 0
    epochs = config.get('epochs', 20)
    deadline = start + seconds
    stop_reason = 'epoch_limit'
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen, completed = 0., 0, True
        epoch_start = time.monotonic()
        for batch in loader:
            if time.monotonic() >= deadline:
                completed = False
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['target'].float())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(batch['target'])
            seen += len(batch['target'])
        if not completed:
            stop_reason = 'time_budget'
            break
        report = {'epoch': epoch, 'loss': total / max(seen, 1), 'train_seconds': time.monotonic()-epoch_start}
        predictions = None
        if not final:
            eval_start = time.monotonic()
            predictions = probabilities(model, dev_dataset)
            ranking = evaluation.rank(predictions)
            score, _ = metrics(ranking, evaluation.candidates)
            report.update(score)
            report['evaluation_seconds'] = time.monotonic() - eval_start
            improved = report['mrr25'] > best + 1e-8
        else:
            improved = True
        if improved:
            best = report.get('mrr25', 0.)
            best_epoch, stale = epoch, 0
            checkpoint = {'architecture': architecture, 'metadata_dim': metadata_dim,
                          'preprocessing': prep, 'config': config, 'epoch': epoch,
                          'state_dict': model.state_dict(), 'data_manifest': json.loads((root / 'manifest.json').read_text())}
            torch.save(checkpoint, output / 'model.pt.tmp')
            (output / 'model.pt.tmp').replace(output / 'model.pt')
            if predictions is not None:
                np.save(output / 'dev_probabilities.npy', predictions)
        else:
            stale += 1
        history.append(report)
        write_json(output / 'history.json', history)
        print(architecture, seed, json.dumps(report), flush=True)
        if not final and stale >= 3:
            stop_reason = 'early_stopping'
            break
    if not best_epoch:
        raise RuntimeError('Budget was too small to complete one epoch; no usable checkpoint was written')
    result = {'architecture': architecture, 'seed': seed, 'best_epoch': best_epoch,
              'best_mrr25': best if not final else None, 'seconds': time.monotonic()-start,
              'parameters': sum(p.numel() for p in model.parameters()), 'spectra': len(dataset),
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'stop_reason': stop_reason, 'final_refit': final, 'checkpoint': str(output / 'model.pt')}
    write_json(output / 'result.json', result)
    return result
