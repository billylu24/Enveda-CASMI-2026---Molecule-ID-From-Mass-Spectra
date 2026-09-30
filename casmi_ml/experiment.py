"""Run staged experiments and freeze one Kaggle selection without leaderboard tuning."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.data import prepare, write_json
from casmi_ml.models import SpectrumDataset
from casmi_ml.ranking import metrics, rrf
from casmi_ml.training import (
    benchmark,
    cached,
    configure,
    load_checkpoint,
    load_evaluation,
    probabilities,
    train,
)


def choose_models(results, count=2):
    remaining, selected = list(results), []
    while remaining and len(selected) < count:
        best = max(r['best_mrr25'] for r in remaining)
        eligible = [r for r in remaining if r['best_mrr25'] >= best - .005]
        winner = min(eligible, key=lambda r: (r['seconds'], r['parameters'], r['architecture']))
        selected.append(winner)
        remaining.remove(winner)
    return selected


def fused(evaluation, neural_rankings, weight):
    result = {}
    for key in evaluation.groups:
        if not neural_rankings:
            result[key] = evaluation.baseline[key]
        else:
            neural = rrf([r[key] for r in neural_rankings])
            result[key] = rrf([evaluation.baseline[key], neural], [1-weight, weight])
    return result


def bootstrap_difference(first, baseline, repetitions=2000):
    joined = first.set_index('key').join(baseline.set_index('key'), lsuffix='_model', rsuffix='_baseline')
    differences = (joined.reciprocal_rank_model - joined.reciprocal_rank_baseline).to_numpy()
    rng = np.random.default_rng(42)
    samples = [float(rng.choice(differences, len(differences), replace=True).mean()) for _ in range(repetitions)]
    return {'difference': float(differences.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist()}


def evaluate_selection(root, selection, split, coconut):
    root = Path(root)
    marker = root / f'{split}_report.json'
    if marker.exists():
        return json.loads(marker.read_text())
    evaluation = load_evaluation(root, split, coconut)
    rankings = []
    for path in selection['checkpoints']:
        model, checkpoint = load_checkpoint(path)
        dataset = SpectrumDataset(cached(root, split, checkpoint['preprocessing']))
        ranking = evaluation.rank(probabilities(model, dataset))
        rankings.append(ranking)
    result_rankings = fused(evaluation, rankings, selection['neural_weight'])
    report, per_molecule = metrics(result_rankings, evaluation.candidates)
    baseline_report, baseline_rows = metrics(evaluation.baseline, evaluation.candidates)
    report['baseline'] = baseline_report
    report['vs_baseline'] = bootstrap_difference(per_molecule, baseline_rows)
    per_molecule.to_csv(root / f'{split}_per_molecule.csv', index=False)
    baseline_rows.to_csv(root / f'{split}_baseline_per_molecule.csv', index=False)
    write_json(marker, report)
    return report


def run(root, coconut, config_path, data_dir="data"):
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    configure(threads=config['threads'])
    root.mkdir(parents=True, exist_ok=True)
    config_snapshot = root / 'experiment_config.json'
    if config_snapshot.exists() and json.loads(config_snapshot.read_text()) != config:
        raise ValueError('Experiment configuration changed; use a new root')
    write_json(config_snapshot, config)
    if not (root / 'manifest.json').exists():
        prepare(Path(data_dir) / 'train.parquet', root)
    budget_path = root / 'budget.json'
    if not budget_path.exists():
        write_json(budget_path, {'limit_seconds': config['budget_hours'] * 3600, 'used_seconds': 0.})
    budget = json.loads(budget_path.read_text())
    def account(seconds):
        budget['used_seconds'] += seconds
        write_json(budget_path, budget)
    def remaining():
        return max(0., budget['limit_seconds'] - budget['used_seconds'])
    benchmark_path = root / 'benchmark.json'
    if benchmark_path.exists():
        speed = json.loads(benchmark_path.read_text())
    else:
        start = time.monotonic()
        reservation = min(3600., remaining())
        if reservation <= 0:
            raise RuntimeError('Training budget exhausted before benchmark')
        account(reservation)
        try:
            speed = benchmark(root, config['benchmark_seconds_per_model'], config['threads'])
        finally:
            account(time.monotonic() - start - reservation)
    evaluation = load_evaluation(root, 'dev', coconut)
    baseline_report, _ = metrics(evaluation.baseline, evaluation.candidates)
    write_json(root / 'baseline_dev.json', baseline_report)
    results = []
    def run_one(architecture, seed, seconds, final=False, epochs=None):
        out = root / 'runs' / (f'{architecture}_{seed}' + ('_final' if final else ''))
        old = (out / 'result.json').exists()
        start = time.monotonic()
        slot = min(seconds, remaining())
        if not old:
            if slot <= 0:
                raise RuntimeError('Training budget exhausted')
            # Reserve the full slot before launch. A hard-killed job therefore
            # cannot silently reclaim its budget when the experiment resumes.
            account(slot)
        try:
            return train(root, {'architecture': architecture, 'seed': seed,
                         'epochs': epochs or config['epochs'], 'lr': config['lr']}, out, coconut,
                         evaluation=evaluation, seconds=slot, final=final,
                         train_fraction=1. if final else speed['common_train_fraction'], threads=config['threads'])
        finally:
            if not old:
                account(time.monotonic() - start - slot)
    for architecture in config['architectures']:
        results.append(run_one(architecture, 42, config['screen_seconds_per_model']))
    shortlisted = choose_models(results)
    for item in shortlisted:
        for seed in config['repeat_seeds']:
            results.append(run_one(item['architecture'], seed, config['repeat_seconds_per_model']))
    pd.DataFrame(results).to_csv(root / 'experiments.csv', index=False)
    summaries = []
    for item in shortlisted:
        runs = [r for r in results if r['architecture'] == item['architecture']]
        summaries.append({'architecture': item['architecture'],
                          'mean_mrr25': float(np.mean([r['best_mrr25'] for r in runs])),
                          'std_mrr25': float(np.std([r['best_mrr25'] for r in runs]))})
    write_json(root / 'seed_summary.json', summaries)
    # Architecture choice uses mean across seeds; inference uses fixed seed 42.
    best_arch = max(summaries, key=lambda r: r['mean_mrr25'])['architecture']
    singles = [r for r in shortlisted if r['architecture'] == best_arch]
    groups = [[], singles, shortlisted]
    comparisons = []
    for members in groups:
        neural = [evaluation.rank(np.load(Path(r['checkpoint']).parent / 'dev_probabilities.npy')) for r in members]
        for weight in ([0.] if not members else [.25, .5, .75, 1.]):
            ranking = fused(evaluation, neural, weight)
            report, _ = metrics(ranking, evaluation.candidates)
            comparisons.append({**report, 'neural_weight': weight,
                                'architectures': [r['architecture'] for r in members],
                                'checkpoints': [r['checkpoint'] for r in members],
                                'parameters': sum(r['parameters'] for r in members)})
    best = max(r['mrr25'] for r in comparisons)
    selection = min([r for r in comparisons if r['mrr25'] >= best - .005],
                    key=lambda r: (len(r['checkpoints']), r['parameters'], -r['mrr25']))
    selection['selection_basis'] = 'development MRR@25; prefer simpler within 0.005; holdout not used for selection'
    selection['rules_status'] = 'Current Kaggle rules not reverified: browser access was denied.'
    write_json(root / 'dev_comparisons.json', comparisons)
    frozen_path = root / 'selection.json'
    if frozen_path.exists():
        old = json.loads(frozen_path.read_text())
        if old != selection:
            raise RuntimeError('Frozen selection differs. Use a fresh experiment directory instead of tuning the holdout.')
    else:
        write_json(frozen_path, selection)
    holdout = evaluate_selection(root, selection, 'holdout', coconut)
    diagnostic = evaluate_selection(root, selection, 'diagnostic', coconut)
    # Refit each selected architecture with a shared three-hour cap. Keep the
    # independently evaluated checkpoint if refit cannot finish an epoch.
    final_selection = dict(selection)
    final_selection['validation_checkpoints'] = list(selection['checkpoints'])
    if selection['checkpoints']:
        final_frame = pd.concat([pd.read_parquet(root / 'train.parquet'), pd.read_parquet(root / 'dev.parquet')], ignore_index=True)
        final_frame.to_parquet(root / 'final_train.parquet', index=False)
        final_paths = []
        per_model_seconds = min(config['final_seconds'], remaining()) / len(selection['checkpoints'])
        for architecture, fallback in zip(selection['architectures'], selection['checkpoints']):
            epochs = max(1, int(np.median([r['best_epoch'] for r in results if r['architecture'] == architecture])))
            if per_model_seconds <= 0:
                final_paths.append(fallback)
                continue
            try:
                fitted = run_one(architecture, 42, per_model_seconds, final=True, epochs=epochs)
            except RuntimeError as error:
                if 'Budget was too small to complete one epoch' not in str(error):
                    raise
                final_paths.append(fallback)
            else:
                final_paths.append(fitted['checkpoint'])
        final_selection['checkpoints'] = final_paths
    final_selection['refit_note'] = 'Final refit uses train+dev; held-out metrics describe frozen pre-refit checkpoints.'
    write_json(root / 'final_selection.json', final_selection)
    report = '# CASMI experiment results\n\n'
    report += f"Recommended: {selection['architectures'] or ['retrieval baseline']}; neural RRF weight {selection['neural_weight']}.\n\n"
    report += f"Development MRR@25: {selection['mrr25']:.4f}. Holdout MRR@25: {holdout['mrr25']:.4f}; baseline: {holdout['baseline']['mrr25']:.4f}.\n\n"
    report += f"Holdout difference and 95% bootstrap interval: {holdout['vs_baseline']}.\n\n"
    report += f"Diagnostic MRR@25: {diagnostic['mrr25']:.4f}. Budget accounted: {budget['used_seconds']/3600:.2f} hours.\n\n"
    report += 'Local metrics are not Kaggle leaderboard scores. Current competition rules have not been reverified.\n'
    (root / 'REPORT.md').write_text(report)
    from casmi_ml.known_audit import audit
    audit(root, coconut)
    deployment_gate(root)
    from casmi_ml.guarded import run as run_guarded
    run_guarded(root, coconut)
    from casmi_ml.report import render
    render(root)
    print(report, flush=True)
    return final_selection



def deployment_gate(root):
    """Reject a frozen candidate that fails the plan's held-out improvement check.

    This is an accept/reject check, not a search over alternative models on holdout.
    The frozen development choice and its independent metrics remain unchanged.
    """
    root = Path(root)
    final_path = root / 'final_selection.json'
    final = json.loads(final_path.read_text())
    if final.get('deployment_mode') == 'confidence_guarded':
        guarded_report = json.loads((root / 'guarded_fresh_report.json').read_text())
        if not guarded_report['accepted']:
            raise RuntimeError('Confidence routing does not have a passing fresh holdout report')
        return final
    if final.get('deployment_mode') == 'historical_hybrid':
        return final
    heldout = json.loads((root / 'holdout_report.json').read_text())
    known_path = root / 'known_spectrum_report.json'
    known_failure = known_path.exists() and not json.loads(known_path.read_text())['accepted']
    if (final['checkpoints'] and heldout['mrr25'] <= heldout['baseline']['mrr25']) or known_failure:
        write_json(root / 'research_selection.json', final)
        final['rejected_checkpoints'] = final['checkpoints']
        final['checkpoints'] = []
        final['architectures'] = []
        final['neural_weight'] = 0.
        final['deployment_mode'] = 'historical_hybrid'
        final['deployment_note'] = ('Frozen neural candidate failed known-spectrum compatibility' if known_failure else 'Frozen neural candidate did not beat retrieval on holdout') + '; retain historically submitted hybrid.py. No model or weight retuning.'
        write_json(final_path, final)
    return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'benchmark', 'train', 'evaluate', 'run', 'predict', 'package', 'audit-known', 'guard'])
    parser.add_argument('--root', default='artifacts/experiment')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--coconut', default='external/coconut_structures.parquet')
    parser.add_argument('--config', default='configs/experiments.json')
    parser.add_argument('--architecture', choices=['mlp', 'enhanced', 'metadata', 'deepsets', 'transformer'], default='mlp')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seconds', type=float, default=7200)
    parser.add_argument('--split', choices=['dev', 'holdout', 'diagnostic'], default='dev')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.root)
    if args.command == 'prepare':
        prepare(Path(args.data_dir) / 'train.parquet', root)
    elif args.command == 'benchmark':
        benchmark(root)
    elif args.command == 'train':
        config = json.loads(Path(args.config).read_text())
        train(root, {'architecture': args.architecture, 'seed': args.seed, 'epochs': config['epochs'], 'lr': config['lr']},
              root / 'runs' / f'{args.architecture}_{args.seed}', args.coconut, seconds=args.seconds)
    elif args.command == 'evaluate':
        selection = json.loads((root / 'selection.json').read_text())
        print(evaluate_selection(root, selection, args.split, args.coconut))
    elif args.command == 'run':
        run(root, args.coconut, args.config, args.data_dir)
    elif args.command == 'guard':
        from casmi_ml.guarded import run as run_guarded
        run_guarded(root, args.coconut)
    elif args.command == 'audit-known':
        from casmi_ml.known_audit import audit
        audit(root, args.coconut)
        deployment_gate(root)
    elif args.command == 'predict':
        from casmi_ml.inference import predict
        deployment_gate(root)
        predict(root / 'final_selection.json', args.data_dir, args.coconut, args.output or 'submission_neural.csv')
    elif args.command == 'package':
        from casmi_ml.inference import package
        deployment_gate(root)
        from casmi_ml.report import render
        render(root)
        package(root, args.output or 'kaggle_bundle')


if __name__ == '__main__':
    main()
