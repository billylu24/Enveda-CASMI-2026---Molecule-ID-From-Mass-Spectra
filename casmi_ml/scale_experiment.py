"""GPU capacity/data scaling experiment with frozen routing and fresh acceptance."""
import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from casmi_ml.ablation import route, sha
from casmi_ml.data import COLUMNS, cache_features, fingerprint, write_json
from casmi_ml.experiment import bootstrap_difference
from casmi_ml.ranking import build_candidates, metrics, neural_rank
from casmi_ml.scale_models import ScaleModel
from casmi_ml.training import configure

ROOT = Path('artifacts/scale_20260929')
OLD = Path('artifacts/experiment')
ABLATION = Path('artifacts/ablation_20260928')
COCONUT = Path('external/coconut_structures.parquet')
ARCHITECTURES = ['metadata_control', 'wide_metadata', 'wide_enhanced', 'peak_transformer', 'hybrid_transformer']
ROUTING = {'kind': 'free_top1', 'base': 'coconut15', 'threshold': .5, 'weight': .75}


def protocol():
    ROOT.mkdir(parents=True, exist_ok=True)
    spec = {'version': 1, 'architectures': ARCHITECTURES, 'small_molecules': 20000, 'large_molecules': 60000,
            'epochs': 30, 'patience': 5, 'min_epochs': 8, 'lr': .0003, 'seed': 42,
            'loss': 'unweighted BCE', 'optimizer': 'AdamW 1e-4 weight decay; linear 2 epoch warmup then cosine',
            'representation': 'existing train-fitted metadata, 1Da histograms, 64 continuous peak tokens; unchanged candidate pool',
            'routing': ROUTING, 'screening': 'all architectures on identical 20k training molecules; best two non-control by raw dev MRR plus control train on nested 60k',
            'selection': 'maximum routed unknown dev MRR subject to known >= old routed MLP - .001; require improvement over old MLP; single model or top-two probability mean',
            'holdout': 'new 2000 molecule cohort excluding every earlier query cohort; one frozen selection and old MLP controls; no tuning',
            'acceptance': 'routed unknown improvement CI95 lower > 0; known point difference >= -.001',
            'max_run_seconds': 2700, 'gpu_memory_limit_fraction': .60}
    path = ROOT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Frozen protocol changed')
    write_json(path, spec)
    return spec


def sample_keys(keys, seed):
    samples, counts = defaultdict(list), defaultdict(int)
    rng = np.random.default_rng(seed)
    offset = 0
    for batch in pq.ParquetFile('data/train.parquet').iter_batches(batch_size=8192, columns=COLUMNS):
        frame = batch.to_pandas()
        frame['row_id'] = np.arange(offset, offset+len(frame))
        offset += len(frame)
        for row in frame[frame.inchikey14.isin(keys)].to_dict('records'):
            key = row['inchikey14']
            counts[key] += 1
            pos = counts[key]-1 if counts[key] <= 4 else int(rng.integers(counts[key]))
            if pos < 4:
                if len(samples[key]) < 4:
                    samples[key].append(row)
                else:
                    samples[key][pos] = row
        if offset % 409600 == 0:
            print(f'sample scan {offset:,}', flush=True)
    rows = []
    for key in sorted(keys):
        if not samples[key]:
            raise ValueError(f'No spectra for {key}')
        fp = fingerprint(samples[key][0]['normalized_smiles'])
        if fp is None:
            raise ValueError(f'Invalid structure {key}')
        for row in samples[key]:
            row['fingerprint'] = np.packbits(fp).tobytes()
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_large():
    path = ROOT / 'train60k.parquet'
    prep = json.loads((OLD / 'preprocessing.json').read_text())
    if not path.exists():
        catalog = pd.read_parquet(OLD / 'catalog.parquet')
        original = pd.read_parquet(OLD / 'train.parquet')
        keys = set(catalog.loc[catalog.split == 'train', 'inchikey14'].head(60000))
        assert set(original.inchikey14) <= keys
        extra = sample_keys(keys - set(original.inchikey14), 20260929)
        frame = pd.concat([original, extra], ignore_index=True)
        assert frame.inchikey14.nunique() == 60000
        frame.to_parquet(path, index=False)
        write_json(ROOT / 'training_manifest.json', {'molecules': 60000, 'spectra': len(frame),
                   'original_20k_spectra_preserved_exactly': True, 'train_split_only': True,
                   'sha256': sha(path), 'preprocessing': prep})
    cache = ROOT / 'features/train60k'
    if not (cache / 'complete.json').exists():
        cache_features(pd.read_parquet(path), prep, cache)


class SelectedFeatures(Dataset):
    def __init__(self, directory, names, target=True):
        self.names = names + (['target'] if target else [])
        self.arrays = {n: np.load(Path(directory) / f'{n}.npy', mmap_mode='r') for n in self.names}

    def __len__(self):
        return len(next(iter(self.arrays.values())))

    def __getitem__(self, i):
        return {n: torch.from_numpy(np.array(a[i], copy=True)) for n, a in self.arrays.items()}


class Ranker:
    def __init__(self, frame, records, pool):
        self.groups = frame.groupby('inchikey14', sort=True).indices
        self.records = records
        lookup = dict(zip(pool.inchikey14, pool.normalized_smiles))
        self.lookup = lookup
        self.fp_cache = {}
        self.matrices = {}

    def fps(self, keys):
        for k in keys:
            if k not in self.fp_cache:
                fp = fingerprint(self.lookup[k])
                if fp is None:
                    raise ValueError(f'Invalid candidate {k}')
                self.fp_cache[k] = fp
        return np.asarray([self.fp_cache[k] for k in keys], dtype=np.float32).reshape(-1, 2048)

    def neural(self, probabilities, record):
        key = record['key']
        candidates = record['available']['union35']
        if key not in self.matrices:
            self.matrices[key] = self.fps(candidates)
        return neural_rank(probabilities[self.groups[key]].mean(0), candidates, self.matrices[key])

    def raw_mrr(self, probabilities):
        total = 0.
        for r in self.records:
            if r['key'] not in r['available']['union35']:
                continue
            ranking = self.neural(probabilities, r)[:25]
            if r['key'] in ranking:
                total += 1/(ranking.index(r['key'])+1)
        return total/len(self.records)

    def score(self, probabilities, known=False, old=False, historical=False):
        rankings, pools = {}, {}
        for r in self.records:
            key = r['key']
            if known and not r['known']:
                continue
            pool = list(set(r['available']['union35']) | set(r['available']['coconut15']))
            pools[key] = pool
            if historical:
                rankings[key] = r['rankings']['coconut15']
            elif old or r['confidence'] >= .5:
                rankings[key] = route(r, ROUTING)
            elif key not in pool:
                # Truth absent from every candidate: reciprocal rank must be zero.
                rankings[key] = r['rankings']['coconut15']
            else:
                rankings[key] = route({**r, 'neural': self.neural(probabilities, r)}, ROUTING)
        return metrics(rankings, pools)


def dev_rankers():
    frame = pd.read_parquet(OLD / 'dev.parquet')
    catalog = pd.read_parquet(OLD / 'catalog.parquet')
    result = {}
    for mode in ['unknown', 'known']:
        records = json.loads((ABLATION / f'dev_{mode}_records.json').read_text())
        if mode == 'known':
            observed = set(pd.read_parquet(OLD / 'reference/known_dev/rows.parquet', columns=['inchikey14']).inchikey14)
            cat = catalog[((catalog.split == 'train') | catalog.inchikey14.isin(frame.inchikey14)) & catalog.inchikey14.isin(observed)]
            pool = build_candidates(cat, COCONUT, final=True)
        else:
            pool = build_candidates(catalog, COCONUT)
        result[mode] = Ranker(frame, records, pool)
    return result


@torch.inference_mode()
def predict(model, directory, batch_size=256):
    dataset = SelectedFeatures(directory, model.input_names, target=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    output = []
    for batch in loader:
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch)
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def predict_cpu(model, directory, batch_size=128):
    dataset = SelectedFeatures(directory, model.input_names, target=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.cpu().eval()
    return np.concatenate([torch.sigmoid(model(batch)).numpy() for batch in loader])


def train(architecture, size, ranker, spec, seed=42):
    out = ROOT / 'runs' / (f'{architecture}_{size}' + (f'_seed{seed}' if seed != 42 else ''))
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    configure(seed, threads=4)
    prep = json.loads((OLD / 'preprocessing.json').read_text())
    dim = 6 + sum(len(v)+1 for v in prep['categories'].values())
    model = ScaleModel(architecture, dim).cuda()
    names = model.input_names
    directory = OLD / 'features/train' if size == 20000 else ROOT / 'features/train60k'
    dataset = SelectedFeatures(directory, names)
    batch_size = 128 if 'transformer' in architecture else 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                        persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=1e-4, fused=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    best, stale, history = -1., 0, []
    stop = 'epoch_limit'
    for epoch in range(1, spec['epochs']+1):
        if epoch <= 2:
            factor = epoch/2
        else:
            factor = .1 + .9*.5*(1+math.cos(math.pi*(epoch-2)/(spec['epochs']-2)))
        for pg in optimizer.param_groups:
            pg['lr'] = spec['lr'] * factor
        start_epoch = time.monotonic()
        model.train()
        total, seen, complete = 0., 0, True
        for batch in loader:
            if time.monotonic()-started > spec['max_run_seconds']:
                complete = False
                break
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            target = batch.pop('target').float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        if not complete:
            stop = 'time_budget'
            break
        elapsed = time.monotonic()-start_epoch
        probabilities = predict(model, OLD / 'features/dev')
        score = ranker.raw_mrr(probabilities)
        row = {'epoch': epoch, 'loss': total/seen, 'mrr25': score, 'train_seconds': elapsed,
               'total_seconds': time.monotonic()-started, 'lr': spec['lr']*factor}
        if score > best + 1e-8:
            best, stale = score, 0
            checkpoint = {'architecture': architecture, 'metadata_dim': dim, 'preprocessing': prep,
                          'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                          'epoch': epoch, 'training_molecules': size, 'seed': seed, 'protocol': spec}
            torch.save(checkpoint, out / 'model.pt')
            np.save(out / 'dev_probabilities.npy', probabilities)
        else:
            stale += 1
        history.append(row)
        write_json(out / 'history.json', history)
        print(architecture, size, json.dumps(row), flush=True)
        if epoch >= spec['min_epochs'] and stale >= spec['patience']:
            stop = 'early_stopping'
            break
    if best < 0:
        raise RuntimeError('No completed epoch')
    result = {'architecture': architecture, 'seed': seed, 'training_molecules': size, 'training_spectra': len(dataset),
              'parameters': sum(p.numel() for p in model.parameters()), 'best_mrr25': best,
              'best_epoch': max(history, key=lambda r:r['mrr25'])['epoch'], 'epochs_completed': len(history),
              'seconds': time.monotonic()-started, 'gpu_peak_mib': torch.cuda.max_memory_allocated()/2**20,
              'stop_reason': stop, 'checkpoint': str(out/'model.pt')}
    write_json(out/'result.json', result)
    del model, optimizer, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_training(spec):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    torch.cuda.set_per_process_memory_fraction(spec['gpu_memory_limit_fraction'])
    write_json(ROOT/'runtime.json', {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
               'python_pid': os.getpid(), 'numpy': np.__version__})
    rankers = dev_rankers()
    old = {m: rankers[m].score(None, known=m=='known', old=True)[0] for m in rankers}
    write_json(ROOT/'old_dev_baseline.json', old)
    results = [train(a, 20000, rankers['unknown'], spec) for a in ARCHITECTURES]
    finalists = sorted((r for r in results if r['architecture'] != 'metadata_control'),
                       key=lambda r:r['best_mrr25'], reverse=True)[:2]
    write_json(ROOT/'data_scaling_finalists.json', finalists)
    prepare_large()
    for architecture in ['metadata_control'] + [r['architecture'] for r in finalists]:
        results.append(train(architecture, 60000, rankers['unknown'], spec))
    pd.DataFrame(results).to_csv(ROOT/'experiments.csv', index=False)
    comparisons = []
    for r in results:
        probs = np.load(Path(r['checkpoint']).parent/'dev_probabilities.npy')
        item = {'checkpoints': [r['checkpoint']], 'name': f'{r["architecture"]}_{r["training_molecules"]}',
                'parameters': r['parameters']}
        for mode in rankers:
            item[mode] = rankers[mode].score(probs, known=mode=='known')[0]
        comparisons.append(item)
    best_two = sorted(comparisons, key=lambda r:r['unknown']['mrr25'], reverse=True)[:2]
    probabilities = np.mean([np.load(Path(r['checkpoints'][0]).parent/'dev_probabilities.npy') for r in best_two], axis=0)
    ensemble = {'checkpoints': [r['checkpoints'][0] for r in best_two], 'name': 'top_two_probability_mean',
                'parameters': sum(r['parameters'] for r in best_two)}
    for mode in rankers:
        ensemble[mode] = rankers[mode].score(probabilities, known=mode=='known')[0]
    comparisons.append(ensemble)
    for r in comparisons:
        r['eligible'] = r['known']['mrr25'] >= old['known']['mrr25']-.001 and r['unknown']['mrr25'] > old['unknown']['mrr25']
    eligible = [r for r in comparisons if r['eligible']]
    winner = max(eligible, key=lambda r:(r['unknown']['mrr25'], -len(r['checkpoints']), -r['parameters'])) if eligible else None
    write_json(ROOT/'dev_comparisons.json', comparisons)
    selection = {'winner': winner, 'old_baseline': old, 'routing': ROUTING,
                 'frozen_before_fresh_holdout': True, 'protocol_sha256': sha(ROOT/'protocol.json')}
    write_json(ROOT/'selection.json', selection)
    print('FROZEN', json.dumps(selection), flush=True)


def prepare_holdout():
    path = ROOT/'fresh.parquet'
    if path.exists():
        return
    catalog = pd.read_parquet(OLD/'catalog.parquet')
    used = set()
    for directory, name in [(OLD, n) for n in ['train', 'dev', 'holdout', 'fresh_holdout', 'diagnostic', 'final_train']] + [(ABLATION, 'fresh'), (ROOT, 'train60k')]:
        used.update(pd.read_parquet(directory/f'{name}.parquet', columns=['inchikey14']).inchikey14)
    keys = set(catalog.loc[(catalog.split=='holdout') & ~catalog.inchikey14.isin(used), 'inchikey14'].head(2000))
    assert len(keys)==2000 and not keys & used
    frame = sample_keys(keys, 20260930)
    frame.to_parquet(path, index=False)
    write_json(ROOT/'fresh_manifest.json', {'molecules': len(keys), 'spectra': len(frame),
               'keys': sorted(keys), 'overlap_with_previous_queries_or_training': 0, 'sha256': sha(path),
               'seed': 20260930, 'selection_sha256': sha(ROOT/'selection.json')})


def fresh_ranker(mode):
    # Reuse the already tested candidate/reference controls, at this experiment's
    # separate root. Their model is the original metadata_42 checkpoint.
    import casmi_ml.ablation as controls
    previous = controls.ROOT
    controls.ROOT = ROOT
    try:
        records = controls.build_records('fresh', mode)
    finally:
        controls.ROOT = previous
    frame = pd.read_parquet(ROOT/'fresh.parquet')
    catalog = pd.read_parquet(OLD/'catalog.parquet')
    if mode == 'known':
        observed = set(pd.read_parquet(ROOT/'reference/fresh_known/rows.parquet', columns=['inchikey14']).inchikey14)
        catalog = catalog[((catalog.split=='train') | catalog.inchikey14.isin(frame.inchikey14)) & catalog.inchikey14.isin(observed)]
    return Ranker(frame, records, build_candidates(catalog, COCONUT, final=mode=='known'))


def final_evaluation():
    marker = ROOT/'fresh_report.json'
    if marker.exists():
        return json.loads(marker.read_text())
    selection = json.loads((ROOT/'selection.json').read_text())
    winner = selection['winner']
    if winner is None:
        report = {'accepted': False, 'reason': 'No development candidate passed selection; no new holdout exposed.'}
        write_json(marker, report)
        return report
    prepare_holdout()
    collected = []
    configure(42, 4)
    for path in winner['checkpoints']:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = ScaleModel(checkpoint['architecture'], checkpoint['metadata_dim'])
        model.load_state_dict(checkpoint['state_dict'])
        directory = ROOT/'features/fresh'
        if not (directory/'complete.json').exists():
            cache_features(pd.read_parquet(ROOT/'fresh.parquet'), checkpoint['preprocessing'], directory)
        collected.append(predict_cpu(model, directory))
        del model
        torch.cuda.empty_cache()
    probabilities = np.mean(collected, axis=0)
    np.save(ROOT/'selected_fresh_probabilities.npy', probabilities)
    reports = {}
    for mode in ['unknown', 'known']:
        ranker = fresh_ranker(mode)
        candidate, rows = ranker.score(probabilities, known=mode=='known')
        baseline, old_rows = ranker.score(None, known=mode=='known', old=True)
        historical, historical_rows = ranker.score(None, known=mode=='known', historical=True)
        candidate['baseline'] = baseline
        candidate['historical'] = historical
        candidate['vs_previous_neural'] = bootstrap_difference(rows, old_rows)
        candidate['vs_historical'] = bootstrap_difference(rows, historical_rows)
        rows.to_csv(ROOT/f'fresh_{mode}_selected.csv', index=False)
        old_rows.to_csv(ROOT/f'fresh_{mode}_old.csv', index=False)
        historical_rows.to_csv(ROOT/f'fresh_{mode}_historical.csv', index=False)
        reports[mode] = candidate
        del ranker
        gc.collect()
        print('FRESH', mode, json.dumps(candidate), flush=True)
    reports['accepted'] = (reports['unknown']['vs_previous_neural']['ci95'][0] > 0 and
                           reports['known']['vs_previous_neural']['difference'] >= -.001)
    reports['selection'] = winner
    reports['inference_precision'] = 'CPU float32, matching intended offline deployment; training/development screening used GPU BF16'
    write_json(marker, reports)
    return reports


def seed_repeats(spec):
    selection = json.loads((ROOT/'selection.json').read_text())
    if selection['winner'] is None:
        return
    paths = selection['winner']['checkpoints']
    write_json(ROOT/'stability_protocol.json', {
        'selection_sha256': sha(ROOT/'selection.json'), 'seeds': [43, 44],
        'checkpoints_to_repeat': paths,
        'purpose': 'development-only stability diagnosis; does not change frozen inference weights or selection; no fresh-label scoring'})
    rankers = dev_rankers()
    results = []
    for path in paths:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        for seed in [43, 44]:
            result = train(checkpoint['architecture'], checkpoint['training_molecules'], rankers['unknown'], spec, seed)
            probabilities = np.load(Path(result['checkpoint']).parent/'dev_probabilities.npy')
            for mode in rankers:
                result[mode] = rankers[mode].score(probabilities, known=mode=='known')[0]
            results.append(result)
    write_json(ROOT/'seed_repeats.json', results)


def diagnostics():
    """Describe the frozen outcome without changing any selection."""
    report = json.loads((ROOT/'fresh_report.json').read_text())
    if 'unknown' not in report:
        return
    ranker = fresh_ranker('unknown')
    selected = np.load(ROOT/'selected_fresh_probabilities.npy')
    old = np.load(ROOT/'fresh_probabilities.npy')
    raw = {'new_raw_unknown_mrr': ranker.raw_mrr(selected),
           'old_raw_unknown_mrr': ranker.raw_mrr(old),
           'neural_routed_molecules': sum(r['confidence'] < .5 for r in ranker.records),
           'molecules': len(ranker.records)}
    write_json(ROOT/'raw_fresh_diagnostic.json', raw)
    rows = []
    for mode in ['unknown', 'known']:
        records = json.loads((ROOT/f'fresh_{mode}_records.json').read_text())
        confidence = {r['key']: r['confidence'] for r in records}
        new = pd.read_csv(ROOT/f'fresh_{mode}_selected.csv').set_index('key')
        baseline = pd.read_csv(ROOT/f'fresh_{mode}_old.csv').set_index('key')
        new['confidence'] = pd.Series(confidence)
        new['old_rr'] = baseline.reciprocal_rank
        for low, high in [(0,.25),(.25,.5),(.5,.65),(.65,.85),(.85,1.01)]:
            group = new[(new.confidence>=low) & (new.confidence<high)]
            if len(group):
                rows.append({'mode': mode, 'lower': low, 'upper': high, 'molecules': len(group),
                             'selected_mrr': group.reciprocal_rank.mean(), 'old_mrr': group.old_rr.mean()})
    pd.DataFrame(rows).to_csv(ROOT/'confidence_breakdown.csv', index=False)


def render_report(report):
    results = pd.read_csv(ROOT/'experiments.csv')
    comparisons = json.loads((ROOT/'dev_comparisons.json').read_text())
    selection = json.loads((ROOT/'selection.json').read_text())
    text = '# GPU 模型与数据规模实验（2026-09-29）\n\n'
    text += '本轮实际使用 RTX 5070 12GB；单独的 GPU 环境以只读路径复用 CrossView-MedMNIST3D 环境的 CUDA PyTorch，原 CPU 环境保持不变。实际运行版本见 runtime.json。\n\n'
    text += '比赛上一提交已出分 0.169，历史最佳仍为 0.176。以下均为本地代理验证，不是 Kaggle 分数。\n\n'
    text += '固定 2048 位指纹、原有 64 峰表示、质量候选池、训练集拟合的预处理和 0.5/0.75 融合规则；高置信度保留完整历史检索，低置信度允许更换首位。先在同一 20K 数据上比较容量及结构，随后对两个开发优胜大模型和小模型对照扩到嵌套的 60K。\n\n'
    text += '| 架构 | 训练分子 | 参数量 | 最佳开发神经 MRR | 最佳轮 / 完成轮 | 秒数 | GPU 峰值 MiB |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in results.itertuples():
        text += f'| {r.architecture} | {r.training_molecules} | {r.parameters:,} | {r.best_mrr25:.6f} | {r.best_epoch}/{r.epochs_completed} | {r.seconds:.1f} | {r.gpu_peak_mib:.0f} |\n'
    text += '\n## 固定融合规则下的开发集\n\n| 模型 | 无参考谱 MRR | 有参考谱 MRR | 合格 |\n|---|---:|---:|---|\n'
    old = selection['old_baseline']
    text += f'| 原已提交 metadata_42 | {old["unknown"]["mrr25"]:.6f} | {old["known"]["mrr25"]:.6f} | 对照 |\n'
    for r in comparisons:
        text += f'| {r["name"]} | {r["unknown"]["mrr25"]:.6f} | {r["known"]["mrr25"]:.6f} | {r["eligible"]} |\n'
    text += f'\n冻结选择：{selection["winner"]["name"] if selection["winner"] else "无新方案"}。独立验收通过：**{report["accepted"]}**。\n\n'
    if 'unknown' in report:
        text += '| 全新留出场景 | 分子数 | 原已提交神经方案 | 新方案 | ΔMRR 95% CI |\n|---|---:|---:|---:|---|\n'
        for mode in ['unknown','known']:
            r = report[mode]
            text += f'| {mode} | {r["molecules"]} | {r["baseline"]["mrr25"]:.6f} | {r["mrr25"]:.6f} | {r["vs_previous_neural"]["ci95"]} |\n'
    text += '\n## 解释限制\n\n'
    text += '- 本轮训练从零开始，使用统一学习率 3e-4、AdamW、两轮预热、余弦下降、BF16；历史旧 MLP 的学习率/调度不同，因此也重新训练了同架构小模型作为本轮公平对照。\n'
    text += '- 最多 30 轮，开发 MRR 连续 5 轮不改善且完成至少 8 轮后早停。结果只描述本次配置，不是架构性能上限；不把参数更多等同于更好。\n'
    text += '- 60K 数据严格保留原 20K 全部样本，额外分子仅来自训练拆分；没有增加开发或留出标签。元数据归一化和词表仍使用原训练集拟合值。\n'
    text += '- 架构/容量筛选为单种子，差异小的结果需要再做多种子确认。候选召回限制仍存在；此轮没有加入质量换算、数据库扩展或新路由器，便于归因。\n'
    text += '- 独立验收与全部旧查询分子及训练分子不重叠；已知谱场景删除查询及相同副本。相对上次神经方案必须未知谱改善区间下限 >0，已知谱点估计下降不超过 0.001。\n'
    text += '- 验收未通过时保留旧部署，不根据此留出结果重新挑模型。未自动提交新模型到 Kaggle。\n'
    text += '- 独立验收使用 CPU float32；开发集筛选使用 GPU BF16。最终精度转换经过真实留出推理，不能用纯 GPU 开发成绩直接代替部署验收。\n'
    repeats_path = ROOT/'seed_repeats.json'
    if repeats_path.exists():
        repeats = json.loads(repeats_path.read_text())
        text += '\n## 已冻结组成模型的额外种子稳定性检查\n\n这些复跑仅使用开发标签，不改变冻结权重，也不在新留出集挑选种子。\n\n| 架构 | 分子数 | 种子 | 神经 MRR | 融合未知 MRR | 融合已知 MRR |\n|---|---:|---:|---:|---:|---:|\n'
        for r in repeats:
            text += f'| {r["architecture"]} | {r["training_molecules"]} | {r["seed"]} | {r["best_mrr25"]:.6f} | {r["unknown"]["mrr25"]:.6f} | {r["known"]["mrr25"]:.6f} |\n'
    raw_path = ROOT/'raw_fresh_diagnostic.json'
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        text += f'\n未知谱图不加保护门槛的纯神经 MRR（只作诊断）：旧模型 {raw["old_raw_unknown_mrr"]:.6f}，新模型 {raw["new_raw_unknown_mrr"]:.6f}。实际融合仅在 {raw["neural_routed_molecules"]}/{raw["molecules"]} 个分子上使用神经分支；分置信区间结果见 confidence_breakdown.csv。该诊断不改变冻结阈值。\n'
    np_path = ROOT/'np_diagnostic.json'
    if np_path.exists():
        np_result = json.loads(np_path.read_text())
        text += '\n## 天然产物来源诊断\n\n'
        text += f'250 个历史天然产物诊断分子，屏蔽同分子参考谱，在相同候选池内比较纯神经排序：旧模型 MRR {np_result["old"]["mrr25"]:.6f} → 新模型 {np_result["new"]["mrr25"]:.6f}；Top-1 {np_result["old"]["top1"]:.2%} → {np_result["new"]["top1"]:.2%}。候选召回均为 98.4%。\n\n'
        text += '该集合在之前实验中已被观察，结果仅作为来源变化的描述性检查，不是新的独立验收，也没有据此修改模型、阈值或权重。\n'
    text += '\n## 复现与候选部署\n\n'
    text += '训练：`.venv-gpu/bin/python -m casmi_ml.scale_experiment train`；固定后的验收：`evaluate`；种子诊断：`repeat`；置信度诊断：`diagnose`。阶段均复用已完成结果，后续调参应使用新根目录和新留出集。\n\n'
    text += '通过验收的 CPU 配置为 `artifacts/scale_20260929/deployment_recipe.json`，推理：`.venv/bin/python -m casmi_ml.secondary_inference --recipe artifacts/scale_20260929/deployment_recipe.json --output submission_scale.csv`。离线包与 Notebook 在 `kaggle_release_scale/`。原已提交版本保持不变，本轮尚未上传或提交。\n'
    (ROOT/'REPORT.md').write_text(text)
    Path('docs/SCALE_20260929.md').write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'repeat', 'diagnose'])
    args = parser.parse_args()
    spec = protocol()
    if args.stage == 'prepare':
        prepare_large()
    elif args.stage == 'train':
        run_training(spec)
    elif args.stage == 'repeat':
        seed_repeats(spec)
    elif args.stage == 'diagnose':
        diagnostics()
        render_report(json.loads((ROOT/'fresh_report.json').read_text()))
    else:
        render_report(final_evaluation())


if __name__ == '__main__':
    main()
