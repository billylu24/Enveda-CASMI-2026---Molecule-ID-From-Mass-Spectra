"""Development-selected confidence routing, validated on an unused holdout cohort."""
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi_ml.data import COLUMNS, fingerprint, write_json
from casmi_ml.experiment import bootstrap_difference, fused
from casmi_ml.models import SpectrumDataset
from casmi_ml.ranking import (
    Evaluation,
    ReferenceIndex,
    build_candidates,
    build_reference,
    metrics,
)
from casmi_ml.training import (
    cached,
    configure,
    load_checkpoint,
    load_evaluation,
    probabilities,
)


def guarded_rank(baseline, neural_fusion, confidence, threshold):
    if confidence >= threshold:
        return baseline
    return baseline[:1] + [k for k in neural_fusion if k not in set(baseline[:1])]


def routed(evaluation, neural, threshold, weight):
    blended = fused(evaluation, neural, weight)
    return {key: guarded_rank(evaluation.baseline[key], blended[key], evaluation.confidence[key], threshold)
            for key in evaluation.groups}


def known_evaluation(root, split, coconut):
    root = Path(root)
    frame = pd.read_parquet(root / f'{split}.parquet')
    catalog = pd.read_parquet(root / 'catalog.parquet')
    # Known-spectrum references may contain queried structures, but never an
    # exact query spectrum (including identical copies elsewhere in the library).
    catalog = catalog[(catalog.split == 'train') | catalog.inchikey14.isin(set(frame.inchikey14))]
    directory = root / 'reference' / f'known_{split}'
    if not (directory / 'complete.json').exists():
        source = json.loads((root / 'manifest.json').read_text())['source']
        build_reference(source, catalog, frame, directory, final=True, exclude_queries=True)
    reference = ReferenceIndex(directory)
    observed = set(reference.rows.inchikey14)
    pool = build_candidates(catalog[catalog.inchikey14.isin(observed)], coconut, final=True)
    evaluation = Evaluation(frame, pool, reference)
    evaluation.known_keys = set(frame.inchikey14) & observed
    return evaluation


def score(evaluation, ranking, known_only=False):
    keys = evaluation.known_keys if known_only else set(ranking)
    report, rows = metrics({k: v for k, v in ranking.items() if k in keys}, evaluation.candidates)
    return report, rows


def fresh_sample(root):
    root = Path(root)
    path = root / 'fresh_holdout.parquet'
    if path.exists():
        return
    catalog = pd.read_parquet(root / 'catalog.parquet')
    used = set(pd.read_parquet(root / 'holdout.parquet').inchikey14)
    keys = set(catalog.loc[(catalog.split == 'holdout') & ~catalog.inchikey14.isin(used), 'inchikey14'].head(2000))
    source = json.loads((root / 'manifest.json').read_text())['source']
    counts, reservoirs = defaultdict(int), defaultdict(list)
    rng = np.random.default_rng(2026)
    offset = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192, columns=COLUMNS):
        frame = batch.to_pandas()
        frame['row_id'] = np.arange(offset, offset+len(frame))
        offset += len(frame)
        for row in frame[frame.inchikey14.isin(keys)].to_dict('records'):
            key = row['inchikey14']
            counts[key] += 1
            pos = counts[key]-1 if counts[key] <= 4 else int(rng.integers(counts[key]))
            if pos < 4:
                if len(reservoirs[key]) < 4:
                    reservoirs[key].append(row)
                else:
                    reservoirs[key][pos] = row
    rows = []
    for key in sorted(keys):
        samples = reservoirs[key]
        if not samples:
            continue
        fp = fingerprint(samples[0]['normalized_smiles'])
        if fp is None:
            continue
        for row in samples:
            row['fingerprint'] = np.packbits(fp).tobytes()
            rows.append(row)
    data = pd.DataFrame(rows)
    for split in ['train', 'dev', 'holdout', 'diagnostic']:
        assert not set(data.inchikey14) & set(pd.read_parquet(root / f'{split}.parquet').inchikey14)
    data.to_parquet(path, index=False)
    write_json(root / 'fresh_holdout_manifest.json', {'molecules': int(data.inchikey14.nunique()),
               'spectra': len(data), 'sampling_seed': 2026, 'excluded_original_holdout': True})


def model_rankings(root, split, evaluation, selection):
    rankings = []
    for path in selection['checkpoints']:
        model, checkpoint = load_checkpoint(path)
        dataset = SpectrumDataset(cached(root, split, checkpoint['preprocessing']))
        rankings.append(evaluation.rank(probabilities(model, dataset)))
    return rankings


def run(root, coconut):
    root = Path(root)
    configure()
    selection = json.loads((root / 'selection.json').read_text())
    weight = selection['neural_weight']
    frozen_path = root / 'guarded_selection.json'
    if not frozen_path.exists():
        unknown = load_evaluation(root, 'dev', coconut)
        unknown_neural = model_rankings(root, 'dev', unknown, selection)
        known = known_evaluation(root, 'dev', coconut)
        known_neural = model_rankings(root, 'dev', known, selection)
        known_base, _ = score(known, known.baseline, True)
        comparisons = []
        for threshold in [.5, .65, .75, .85]:
            unseen_report, _ = score(unknown, routed(unknown, unknown_neural, threshold, weight))
            known_report, _ = score(known, routed(known, known_neural, threshold, weight), True)
            comparisons.append({'threshold': threshold, 'unknown': unseen_report, 'known': known_report,
                                'known_baseline': known_base,
                                'eligible': known_report['mrr25'] >= known_base['mrr25']-.005})
        write_json(root / 'guarded_dev_comparisons.json', comparisons)
        eligible = [r for r in comparisons if r['eligible']]
        if not eligible:
            write_json(frozen_path, {'accepted_development': False, 'reason': 'All gates fail known-spectrum development check'})
            return
        winner = max(eligible, key=lambda r: (r['unknown']['mrr25'], -r['threshold']))
        chosen = {**selection, 'confidence_threshold': winner['threshold'], 'accepted_development': True,
                  'development': winner, 'protocol': 'Full baseline for confidence >= threshold; otherwise protect baseline top1 and fill from frozen neural RRF. Threshold selected only on dev. Independent fresh cohort used for acceptance.'}
        write_json(frozen_path, chosen)
        del unknown, known, unknown_neural, known_neural
        gc.collect()
    chosen = json.loads(frozen_path.read_text())
    if not chosen['accepted_development']:
        return
    # The routing configuration is immutable before these new labels are evaluated.
    fresh_sample(root)
    report_path = root / 'guarded_fresh_report.json'
    if not report_path.exists():
        reports = {}
        for mode in ['unknown', 'known']:
            evaluation = load_evaluation(root, 'fresh_holdout', coconut) if mode == 'unknown' else known_evaluation(root, 'fresh_holdout', coconut)
            neural = model_rankings(root, 'fresh_holdout', evaluation, selection)
            ranking = routed(evaluation, neural, chosen['confidence_threshold'], weight)
            result, rows = score(evaluation, ranking, mode == 'known')
            baseline, baseline_rows = score(evaluation, evaluation.baseline, mode == 'known')
            result['baseline'] = baseline
            result['vs_baseline'] = bootstrap_difference(rows, baseline_rows)
            reports[mode] = result
            rows.to_csv(root / f'guarded_fresh_{mode}.csv', index=False)
            del evaluation, neural, ranking
            gc.collect()
        reports['accepted'] = reports['unknown']['mrr25'] > reports['unknown']['baseline']['mrr25'] and reports['known']['mrr25'] >= reports['known']['baseline']['mrr25']-.005
        write_json(report_path, reports)
    reports = json.loads(report_path.read_text())
    if reports['accepted']:
        research_path = root / 'research_selection.json'
        final = json.loads(research_path.read_text()) if research_path.exists() else json.loads((root / 'final_selection.json').read_text())
        final['deployment_mode'] = 'confidence_guarded'
        final['confidence_threshold'] = chosen['confidence_threshold']
        final['routing_development'] = chosen['development']
        final['deployment_note'] = 'Development-frozen confidence routing passed both unknown and known-spectrum checks on a fresh disjoint holdout cohort.'
        write_json(root / 'final_selection.json', final)
    print(json.dumps({'threshold':chosen['confidence_threshold'], **reports},indent=2),flush=True)
