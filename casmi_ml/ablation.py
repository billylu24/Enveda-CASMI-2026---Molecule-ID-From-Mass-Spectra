"""Historical-hybrid ablations and development-only routing selection.

Run with OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -m casmi_ml.ablation.
Existing models, selections and holdouts are never overwritten.
"""
import gc
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import DataStructs

from baseline import ADDUCT_MASS, PROTON, formula_mass
from casmi_ml.data import COLUMNS, write_json
from casmi_ml.data import fingerprint as array_fingerprint
from casmi_ml.experiment import bootstrap_difference
from casmi_ml.guarded import guarded_rank
from casmi_ml.models import SpectrumDataset
from casmi_ml.ranking import (
    CandidateIndex,
    ReferenceIndex,
    baseline_rank,
    build_candidates,
    build_reference,
    center_mass,
    metrics,
    neural_rank,
    rrf,
)
from casmi_ml.training import cached, configure, load_checkpoint, probabilities
from hybrid import blend, coconut_rank, fingerprint

ROOT = Path('artifacts/ablation_20260928')
OLD = Path('artifacts/experiment')
COCONUT = Path('external/coconut_structures.parquet')
CHECKPOINT = OLD / 'runs/metadata_42/model.pt'
VARIANTS = ['coconut15', 'coconut35', 'union15', 'union35']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def analog_rank(center, library, table, masses, order, cache, ppm, floor):
    if center is None or center <= 0:
        return []
    delta = max(center * ppm * 1e-6, floor)
    ids = order[np.searchsorted(masses, center - delta):
                np.searchsorted(masses, center + delta, side='right')]
    refs = [(fingerprint(smi, cache), score) for _, smi, score in library[:25]]
    refs = [(fp, score) for fp, score in refs if fp is not None]
    if not refs:
        return []
    ref_fps = [fp for fp, _ in refs]
    result = {}
    for row in table.iloc[ids].itertuples():
        fp = fingerprint(row.canonical_smiles, cache)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        score = max(sim * (.5 + .5 * r[1]) for sim, r in zip(sims, refs))
        score *= math.exp(-.5 * ((abs(row.exact_mass - center) / center * 1e6) / 15) ** 2)
        key = row.inchikey[:14]
        if key not in result or score > result[key][1]:
            result[key] = (row.canonical_smiles, score)
    return sorted([(k, smi, s) for k, (smi, s) in result.items()], key=lambda x: -x[2])


def route(record, config):
    baseline = record['rankings'][config['base']]
    if config['kind'] == 'retrieval':
        return baseline
    if config['kind'] == 'current_guard':
        fused = rrf([record['current_full'], record['neural']], [.25, .75])
        return guarded_rank(record['current_full'], fused, record['confidence'], .65)
    # Preserve the exact historical top-25 whenever the gate is closed.
    if record['confidence'] >= config['threshold']:
        return baseline
    if config['kind'] == 'margin' and record['margin'] >= config['margin']:
        return baseline
    extended = baseline + [k for k in record['current_full'] if k not in set(baseline)]
    fused = rrf([extended, record['neural']], [1-config['weight'], config['weight']])
    if config['kind'] == 'protect_top1':
        return baseline[:1] + [k for k in fused if k not in set(baseline[:1])]
    return fused


def configurations():
    configs = [{'name': v, 'base': v, 'kind': 'retrieval'} for v in VARIANTS]
    configs.append({'name': 'current_guard', 'base': 'union35', 'kind': 'current_guard'})
    for threshold in [.35, .5, .65]:
        for weight in [.5, .75]:
            for kind, margin in [('protect_top1', None), ('free_top1', None), ('margin', .05), ('margin', .15)]:
                configs.append({'name': f'{kind}_t{threshold}_w{weight}_m{margin}',
                                'base': 'coconut15', 'kind': kind, 'threshold': threshold,
                                'weight': weight, 'margin': margin})
    return configs


def prepare_protocol():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {'version': 1, 'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha(CHECKPOINT),
                'configs': configurations(), 'dev_molecules': 2000, 'fresh_holdout_molecules': 2000,
                'selection': 'maximize equal-weight known/unknown development MRR, subject to known MRR >= historical hybrid - 0.001 and unknown MRR >= historical hybrid; retrieval simplicity breaks ties',
                'acceptance': 'frozen winner unknown MRR > historical, known MRR >= historical - 0.001, balanced delta > 0',
                'holdout': 'unused holdout keys, deterministic SHA256 order, seed 20260928 reservoir, no retuning',
                'reference_note': 'shared filtered reference index for all variants; historical ranking oracle checked; missing-reference fallback evaluated separately on visible test',
                'coverage': 'development only; extended charge-aware median and union of per-spectrum mass hypotheses; no query labels used to generate masses',
                'public_scores': {'hybrid': .176, 'guarded': .162}}
    path = ROOT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a new experiment directory')
    write_json(path, protocol)
    return protocol


def sample_fresh():
    path = ROOT / 'fresh.parquet'
    if path.exists():
        return
    catalog = pd.read_parquet(OLD / 'catalog.parquet')
    used = set()
    for name in ['train', 'dev', 'holdout', 'diagnostic', 'fresh_holdout', 'final_train']:
        used.update(pd.read_parquet(OLD / f'{name}.parquet', columns=['inchikey14']).inchikey14)
    keys = set(catalog.loc[(catalog.split == 'holdout') & ~catalog.inchikey14.isin(used), 'inchikey14'].head(2000))
    assert len(keys) == 2000 and not keys & used
    rng = np.random.default_rng(20260928)
    counts, samples = defaultdict(int), defaultdict(list)
    offset = 0
    source = json.loads((OLD / 'manifest.json').read_text())['source']
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192, columns=COLUMNS):
        frame = batch.to_pandas()
        frame['row_id'] = np.arange(offset, offset + len(frame))
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
            print(f'fresh sample scan {offset:,}', flush=True)
    rows = []
    for key in sorted(keys):
        fp = array_fingerprint(samples[key][0]['normalized_smiles'])
        if fp is None:
            raise ValueError(f'Invalid fresh structure {key}')
        for row in samples[key]:
            row['fingerprint'] = np.packbits(fp).tobytes()
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_parquet(path, index=False)
    write_json(ROOT / 'fresh_manifest.json', {'molecules': result.inchikey14.nunique(),
               'spectra': len(result), 'keys': sorted(keys), 'overlap_with_prior_queries': 0,
               'seed': 20260928, 'sha256': sha(path)})


def extended_masses(group):
    extra = {'[M+C2H4O2-H]-': (1, formula_mass('C2H4O2') - PROTON),
             '[M+2H]2+': (2, 2 * PROTON), '[M+3H]3+': (3, 3 * PROTON),
             '[M-2H]2-': (2, -2 * PROTON),
             '[M]+': (1, -0.00054858), '[M]-': (1, 0.00054858)}
    masses = []
    for row in group.itertuples():
        if row.adduct in ADDUCT_MASS:
            charge, shift = 1, ADDUCT_MASS[row.adduct]
        elif row.adduct in extra:
            charge, shift = extra[row.adduct]
        else:
            continue
        m = float(row.precursor_mz) * charge - shift
        if np.isfinite(m) and m > 0:
            masses.append(m)
    return masses


def coverage_audit():
    if (ROOT / 'coverage_summary.json').exists():
        return
    frame = pd.read_parquet(OLD / 'dev.parquet')
    coco = pd.read_parquet(COCONUT)
    coco['key'] = coco.inchikey.str[:14]
    by_key = coco.groupby('key').exact_mass.apply(list).to_dict()
    masses = np.sort(coco.exact_mass.to_numpy())
    production = build_candidates(pd.read_parquet(OLD / 'catalog.parquet'), COCONUT).set_index('inchikey14')
    rows = []
    misses = []
    for key, group in frame.groupby('inchikey14'):
        center = center_mass(group)
        extended = extended_masses(group)
        hypotheses = {'current': [] if center is None else [center],
                      'extended_median': [float(np.median(extended))] if extended else [],
                      'extended_union': extended}
        true_masses = by_key.get(key, [])
        row = {'key': key, 'in_coconut': bool(true_masses), 'source': '|'.join(sorted(set(group.ingest_lib))),
               'adducts': '|'.join(sorted(set(group.adduct))), 'current_center': center,
               'formula_mass': formula_mass(group.molecular_formula.iloc[0]),
               'coconut_masses': json.dumps(true_masses), 'extended_masses': json.dumps(extended)}
        for method, centers in hypotheses.items():
            hit = any(abs(c-m) <= max(c*35e-6, .006) for c in centers for m in true_masses)
            row[f'{method}_hit'] = hit
            row[f'{method}_production_hit'] = (key in production.index and
                any(abs(c-float(production.loc[key, 'mass'])) <= max(c*35e-6, .006) for c in centers))
            ids = set()
            for c in centers:
                d = max(c*35e-6, .006)
                ids.update(range(np.searchsorted(masses, c-d), np.searchsorted(masses, c+d, side='right')))
            row[f'{method}_candidate_rows'] = len(ids)
        rows.append(row)
        if true_masses and not row['current_hit']:
            misses.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / 'coverage_per_molecule.csv', index=False)
    pd.DataFrame(misses).to_csv(ROOT / 'mass_window_misses.csv', index=False)
    summary = {'molecules': len(result), 'in_coconut': int(result.in_coconut.sum()),
               'production_deduplicated_pool': {m: int(result[f'{m}_production_hit'].sum())
                                                for m in ['current', 'extended_median', 'extended_union']},
               'duplicate_mass_note': 'Raw COCONUT retains multiple mass/charge forms per 2D key; production deduplicates before mass matching.',
               'methods': {m: {'hits': int(result[f'{m}_hit'].sum()),
                             'recall': float(result[f'{m}_hit'].mean()),
                             'mean_candidate_rows': float(result[f'{m}_candidate_rows'].mean())}
                           for m in ['current', 'extended_median', 'extended_union']},
               'missing_db_sources': result[~result.in_coconut].source.value_counts().head(15).to_dict(),
               'note': 'Coverage-only experiment; larger hypothesis unions may harm ranking. Charged-structure/formula inconsistencies are reported, not corrected with labels.'}
    write_json(ROOT / 'coverage_summary.json', summary)


def build_records(split, mode):
    out = ROOT / f'{split}_{mode}_records.json'
    if out.exists():
        return json.loads(out.read_text())
    data_root = OLD if split == 'dev' else ROOT
    frame = pd.read_parquet(data_root / f'{split}.parquet')
    catalog = pd.read_parquet(OLD / 'catalog.parquet')
    if split == 'dev':
        refdir = OLD / 'reference' / ('known_dev' if mode == 'known' else 'dev')
    else:
        refdir = ROOT / 'reference' / f'{split}_{mode}'
    if not (refdir / 'complete.json').exists():
        allowed = catalog if mode == 'unknown' else catalog[(catalog.split == 'train') | catalog.inchikey14.isin(frame.inchikey14)]
        source = json.loads((OLD / 'manifest.json').read_text())['source']
        build_reference(source, allowed, frame, refdir, final=mode == 'known', exclude_queries=mode == 'known')
    reference = ReferenceIndex(refdir)
    observed = set(reference.rows.inchikey14)
    if mode == 'known':
        catalog = catalog[((catalog.split == 'train') | catalog.inchikey14.isin(frame.inchikey14)) & catalog.inchikey14.isin(observed)]
    pool = build_candidates(catalog, COCONUT, final=mode == 'known')
    index = CandidateIndex(pool)
    coco = pd.read_parquet(COCONUT, columns=['inchikey', 'canonical_smiles', 'exact_mass'])
    union = index.catalog.rename(columns={'inchikey14': 'inchikey', 'normalized_smiles': 'canonical_smiles', 'mass': 'exact_mass'})
    tables = {'coconut': coco, 'union': union}
    orders = {name: np.argsort(t.exact_mass.to_numpy()) for name, t in tables.items()}
    sorted_mass = {name: t.exact_mass.to_numpy()[orders[name]] for name, t in tables.items()}
    smiles = reference.rows.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles.to_dict()
    model, checkpoint = load_checkpoint(CHECKPOINT)
    probpath = ROOT / f'{split}_probabilities.npy'
    if not probpath.exists():
        probs = probabilities(model, SpectrumDataset(cached(data_root, split, checkpoint['preprocessing'])))
        np.save(probpath, probs)
    probs = np.load(probpath)
    del model
    cache, records = {}, []
    for i, (key, ids) in enumerate(frame.groupby('inchikey14', sort=True).indices.items()):
        group = frame.iloc[ids]
        center = center_mass(group)
        library = [(k, smiles[k], s) for k, s in reference.rank(group, center)]
        candidates, fps = index.fps(index.query(center))
        keys = candidates.inchikey14.tolist()
        current = baseline_rank(group, candidates, fps, reference, center)
        rankings, available = {}, {}
        for variant in VARIANTS:
            name = 'coconut' if variant.startswith('coconut') else 'union'
            ppm = 15 if variant.endswith('15') else 35
            analog = analog_rank(center, library, tables[name], sorted_mass[name], orders[name], cache, ppm, .004 if ppm == 15 else .006)
            if variant == 'coconut15':
                oracle = coconut_rank(center, library, coco, sorted_mass[name], orders[name], cache) if center is not None else []
                assert analog == oracle, (split, mode, key, 'historical analog mismatch')
            rankings[variant] = [k for k, _ in blend(library, analog)]
            available[variant] = list(dict.fromkeys([k for k, _, _ in library] + [k for k, _, _ in analog]))
        # union35 uses the production implementation, including key tie-breaking
        # and filling candidates when no reference fingerprint is available.
        rankings['union35'] = current[:25]
        available['union35'] = keys
        confidence = library[0][2] if library else 0.
        margin = confidence - library[1][2] if len(library) > 1 else confidence
        records.append({'key': key, 'known': key in observed, 'rankings': rankings,
                        'available': available, 'current_full': current,
                        'neural': neural_rank(probs[ids].mean(0), keys, fps),
                        'confidence': confidence, 'margin': margin, 'candidate_count': len(keys)})
        if (i+1) % 250 == 0:
            print(f'{split}/{mode}: {i+1} molecules', flush=True)
    write_json(out, records)
    del reference, index, cache, probs
    gc.collect()
    return records


def evaluate(records, config, mode):
    chosen = [r for r in records if mode != 'known' or r['known']]
    rankings = {r['key']: route(r, config) for r in chosen}
    pools = {r['key']: r['available'][config['base']] if config['kind'] == 'retrieval'
             else list(set(r['available'][config['base']]) | set(r['available']['union35'])) for r in chosen}
    return metrics(rankings, pools)


def develop(protocol):
    frozen = ROOT / 'selection.json'
    if frozen.exists():
        return json.loads(frozen.read_text())
    records = {mode: build_records('dev', mode) for mode in ['unknown', 'known']}
    results = []
    for config in protocol['configs']:
        result = {'config': config}
        for mode, mode_records in records.items():
            report, rows = evaluate(mode_records, config, mode)
            result[mode] = report
            rows.to_csv(ROOT / f'dev_{mode}_{config["name"]}.csv', index=False)
        results.append(result)
    baseline = results[0]
    for r in results:
        r['eligible'] = (r['known']['mrr25'] >= baseline['known']['mrr25'] - .001 - 1e-12 and
                         r['unknown']['mrr25'] >= baseline['unknown']['mrr25'] - 1e-12)
        r['balanced_mrr'] = (r['known']['mrr25'] + r['unknown']['mrr25']) / 2
    winner = max((r for r in results if r['eligible']),
                 key=lambda r: (r['balanced_mrr'], r['config']['kind'] == 'retrieval'))
    write_json(ROOT / 'dev_comparisons.json', results)
    write_json(frozen, {'winner': winner, 'baseline': baseline, 'protocol_sha256': sha(ROOT / 'protocol.json'),
                       'checkpoint_sha256': sha(CHECKPOINT), 'frozen_before_fresh_evaluation': True})
    print('FROZEN WINNER', json.dumps(winner), flush=True)
    return json.loads(frozen.read_text())


def final_evaluation(selection, protocol):
    path = ROOT / 'fresh_report.json'
    if path.exists():
        return json.loads(path.read_text())
    sample_fresh()
    selected = selection['winner']['config']
    controls = protocol['configs'][:5]
    configs = controls + ([] if selected['name'] in {c['name'] for c in controls} else [selected])
    secondary_path = ROOT / 'secondary_selection.json'
    secondary = json.loads(secondary_path.read_text())['winner']['config'] if secondary_path.exists() else None
    if secondary and secondary['name'] not in {c['name'] for c in configs}:
        configs.append(secondary)
    results = {c['name']: {'config': c} for c in configs}
    for mode in ['unknown', 'known']:
        records = build_records('fresh', mode)
        _, base_rows = evaluate(records, controls[0], mode)
        for c in configs:
            report, rows = evaluate(records, c, mode)
            report['vs_historical'] = bootstrap_difference(rows, base_rows)
            rows.to_csv(ROOT / f'fresh_{mode}_{c["name"]}.csv', index=False)
            results[c['name']][mode] = report
        del records
        gc.collect()
    winner = results[selected['name']]
    du = winner['unknown']['vs_historical']['difference']
    dk = winner['known']['vs_historical']['difference']
    result = {'selection': selected, 'results': results,
              'accepted': du > 0 and dk >= -.001 and (du + dk) > 0,
              'balanced_delta': (du+dk)/2, 'note': 'No selection or retuning from these results. Equal-weight mixture is an explicit proxy, not the unknown competition class distribution.'}
    if secondary:
        sr = results[secondary['name']]
        su, sk = sr['unknown']['vs_historical'], sr['known']['vs_historical']
        result['secondary_selection'] = secondary
        result['secondary_accepted'] = su['difference'] > 0 and sk['difference'] >= -.001 and su['ci95'][0] > 0
    write_json(path, result)
    return result


def report_results(final):
    dev = json.loads((ROOT / 'dev_comparisons.json').read_text())
    coverage = json.loads((ROOT / 'coverage_summary.json').read_text())
    text = '# Hybrid 与神经融合消融实验（2026-09-28）\n\n'
    text += 'Kaggle 已确认：历史 hybrid 0.176，上一版 guarded MLP 0.162。以下为本地代理指标，不能当作榜单分数。\n\n'
    text += '本轮结论：原 hybrid 候选来源应作为新实验的稳定对照。未知谱图优先的次要冻结方案通过独立验收：检索置信度 ≥0.5 保留历史完整排序；否则采用神经 RRF 权重 0.75，并允许替换首位。主选择仍是上一版保护式融合，但它未通过本轮未知谱图必须改善的验收。两个选择均保留原始记录，不根据留出结果重新命名胜者。\n\n'
    text += '次要方案相对历史候选规则改善未知谱图，已知谱图基本持平；上一版共同候选方案在已知谱图场景仍更强。因此这是场景取舍，而不是对所有场景的全面提升。正式推理配置与 Kaggle 提交未修改。\n\n'
    text += '开发集选择后冻结方案；全新 2000 分子留出集与所有旧查询集合及神经训练集不重叠。已知谱图场景排除查询及相同谱图副本，仅评估确有其他参考谱图的分子。\n\n'
    visible_path = ROOT / 'visible_reproduction.json'
    if visible_path.exists():
        visible = json.loads(visible_path.read_text())
        text += '## 可见样本复现\n\n'
        text += '| 方案 | 与历史完整顺序一致 /400 | 平均候选结构重合数 |\n|---|---:|---:|\n'
        for name, r in visible['variants'].items():
            text += f'| {name} | {r["same_order_historical"]} | {r["mean_overlap_historical"]:.2f} |\n'
        text += '\n原始参考库加载器复现历史 hybrid 的 400/400 完整结构顺序；共同候选版与上一版实际输出为 399/400 一致，仍存在一例排序差异，故不能称为完整逐位复现。候选来源改变比单独扩窗影响更多列表；没有真实答案，不能据此给出榜单降分的因果比例。\n\n'
        routing_path = ROOT / 'visible_routing.json'
        if routing_path.exists():
            vr = json.loads(routing_path.read_text())
            text += f'可见样本最低匹配置信度 {vr["minimum_confidence"]:.8f}，两个冻结方案都不会触发神经分支。这些样本主要用于功能一致性检查。\n\n'
    text += '## 开发集比较\n\n'
    text += '| 开发方案 | 无参考谱 MRR | 有参考谱 MRR | 无参考谱 Top-1 | 合格 |\n|---|---:|---:|---:|---|\n'
    for r in dev:
        text += f'| {r["config"]["name"]} | {r["unknown"]["mrr25"]:.6f} | {r["known"]["mrr25"]:.6f} | {r["unknown"]["top1"]:.4%} | {r["eligible"]} |\n'
    text += f'\n冻结方案：`{final["selection"]["name"]}`。独立验收通过：**{final["accepted"]}**。\n\n'
    if 'secondary_selection' in final:
        text += f'在读取全新留出结果之前，另冻结未知谱图优先方案（同一开发验收约束）：`{final["secondary_selection"]["name"]}`；次要终点验收：**{final["secondary_accepted"]}**。这不替换主选择，冻结记录见 secondary_selection.json。\n\n'
    text += '| 全新留出方案 | 场景 | 分子数 | 候选召回 | MRR | Top-1 | Top-5 | Top-25 | 相对历史 ΔMRR（95% CI） |\n|---|---|---:|---:|---:|---:|---:|---:|---|\n'
    for name, item in final['results'].items():
        for mode in ['unknown', 'known']:
            r = item[mode]
            d = r['vs_historical']
            text += f'| {name} | {mode} | {r["molecules"]} | {r["candidate_recall"]:.2%} | {r["mrr25"]:.6f} | {r["top1"]:.2%} | {r["top5"]:.2%} | {r["top25"]:.2%} | {d["difference"]:+.6f} [{d["ci95"][0]:+.6f}, {d["ci95"][1]:+.6f}] |\n'
    text += '\n## 质量换算与候选覆盖（开发集）\n\n'
    text += f'2000 个开发分子中 {coverage["in_coconut"]} 个存在于 COCONUT。\n\n'
    for name, r in coverage['methods'].items():
        text += f'- {name}: 命中 {r["hits"]}；召回 {r["recall"]:.2%}；平均候选记录数 {r["mean_candidate_rows"]:.1f}。\n'
    text += f'\n使用当前按结构键预先去重的候选池时，命中数分别为：{coverage["production_deduplicated_pool"]}。原始 COCONUT 中同一二维键可能有不同质量/电荷记录，提前去重也会损失召回。\n'
    text += '\n扩展换算覆盖双/三电荷、分子离子和乙酸加合物；union 使用各谱图推导质量的并集，完全不使用答案确定候选窗口。该部分仅验证召回，尚不能证明排序收益。数据库缺失与同结构键质量/电荷不一致的记录保留在 coverage_per_molecule.csv 与 mass_window_misses.csv。\n\n'
    text += '## 解释边界\n\n'
    text += '- coconut15 是历史候选来源与窗口；coconut35 仅扩窗；union15 改为共同候选来源；union35 是上一版共同候选检索实现。扩窗同时包含绝对质量下限从 0.004 到 0.006 Da 的变化。\n'
    text += '- 四个检索方案共享参考谱图索引；历史候选打分逐分子与原 coconut_rank 对照，参考库预筛及无谱图回退与完整历史提交的差异另见 visible_reproduction.json。\n'
    text += '- 神经模型固定为 metadata_42，未用 train+dev 重训权重评估开发集。保护分支保留历史完整 Top-25；低置信度允许首位替换或结合首二名分差。\n'
    text += '- 选择要求已知谱 MRR 下降不超过 0.001，未知谱不下降，在合格方案中最大化两场景等权平均。比赛真实类别比例未知，等权指标只是代理。\n'
    text += '- 留出集只做冻结方案验收及预声明消融对照；不得根据本表再选择另一配置并声称它是独立验收的胜者。\n'
    text += '- 当前实验未扩充外部结构库、未重新训练更大模型、未提交 Kaggle。\n'
    mix_path = ROOT / 'mixture_sensitivity.json'
    if mix_path.exists():
        mix = json.loads(mix_path.read_text())
        fraction = mix['unknown_fraction_break_even']
        if fraction is not None:
            text += f'\n两冻结方案在本地代理分布的线性混合中，未知谱图比例的收益交叉点约为 {fraction:.1%}。这不是比赛类别比例的估计；来源分层数据见 fresh_source_breakdown.csv。\n'
    (ROOT / 'REPORT.md').write_text(text)
    Path('docs/ABLATION_20260928.md').write_text(text)


def add_secondary(final):
    """Evaluate the separately development-frozen sensitivity endpoint on cached predictions."""
    path = ROOT / 'secondary_selection.json'
    if not path.exists() or 'secondary_selection' in final:
        return final
    selection = json.loads(path.read_text())
    config = selection['winner']['config']
    result = {'config': config}
    baseline = configurations()[0]
    for mode in ['unknown', 'known']:
        records = build_records('fresh', mode)
        report, rows = evaluate(records, config, mode)
        _, baseline_rows = evaluate(records, baseline, mode)
        report['vs_historical'] = bootstrap_difference(rows, baseline_rows)
        rows.to_csv(ROOT / f'fresh_{mode}_{config["name"]}.csv', index=False)
        result[mode] = report
    su, sk = result['unknown']['vs_historical'], result['known']['vs_historical']
    final['results'][config['name']] = result
    final['secondary_selection'] = config
    final['secondary_accepted'] = su['difference'] > 0 and sk['difference'] >= -.001 and su['ci95'][0] > 0
    write_json(ROOT / 'fresh_report.json', final)
    return final


def freeze_secondary():
    path = ROOT / 'secondary_selection.json'
    if path.exists():
        return
    if (ROOT / 'fresh_unknown_records.json').exists():
        raise RuntimeError('Cannot introduce a secondary endpoint after fresh predictions exist')
    comparisons = json.loads((ROOT / 'dev_comparisons.json').read_text())
    winner = max((r for r in comparisons if r['eligible']), key=lambda r: r['unknown']['mrr25'])
    write_json(path, {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'winner': winner,
                     'selection_basis': 'Maximum unknown development MRR among predeclared eligible configurations; secondary sensitivity endpoint',
                     'frozen_before_fresh_predictions': True,
                     'acceptance': 'unknown MRR > historical; known MRR >= historical - 0.001; unknown delta CI lower bound > 0'})


def source_diagnostics(final):
    """Descriptive breakdown only; never changes either frozen selection."""
    frame = pd.read_parquet(ROOT / 'fresh.parquet', columns=['inchikey14', 'ingest_lib'])
    sources = frame.groupby('inchikey14').ingest_lib.agg(lambda s: '|'.join(sorted(set(s))))
    output = []
    for mode in ['unknown', 'known']:
        for name in final['results']:
            rows = pd.read_csv(ROOT / f'fresh_{mode}_{name}.csv').set_index('key')
            rows['source'] = sources
            for source, group in rows.groupby('source'):
                output.append({'mode': mode, 'variant': name, 'source': source,
                               'molecules': len(group), 'mrr25': group.reciprocal_rank.mean(),
                               'recall': group.covered.mean(), 'top1': group.top1.mean()})
    pd.DataFrame(output).to_csv(ROOT / 'fresh_source_breakdown.csv', index=False)
    primary = final['results'][final['selection']['name']]
    secondary = final['results'][final['secondary_selection']['name']]
    du = secondary['unknown']['mrr25'] - primary['unknown']['mrr25']
    dk = secondary['known']['mrr25'] - primary['known']['mrr25']
    crossover = -dk / (du-dk) if du != dk else None
    write_json(ROOT / 'mixture_sensitivity.json', {
        'secondary_minus_primary_unknown': du, 'secondary_minus_primary_known': dk,
        'unknown_fraction_break_even': crossover if crossover is not None and 0 <= crossover <= 1 else None,
        'note': 'Descriptive linear mixture of conditional proxy MRR; not an estimate of competition proportions or expected Kaggle score.'})


def main():
    configure(threads=4)
    protocol = prepare_protocol()
    coverage_audit()
    selection = develop(protocol)
    freeze_secondary()
    final = add_secondary(final_evaluation(selection, protocol))
    source_diagnostics(final)
    report_results(final)
    print(json.dumps({'selection': final['selection'], 'accepted': final['accepted']}), flush=True)


if __name__ == '__main__':
    main()
