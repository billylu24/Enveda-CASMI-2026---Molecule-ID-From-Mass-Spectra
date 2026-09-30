"""Acceptance audit for known-spectrum compatibility; never tunes model weights."""
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.experiment import bootstrap_difference, fused
from casmi_ml.models import SpectrumDataset
from casmi_ml.ranking import (
    Evaluation,
    ReferenceIndex,
    build_candidates,
    build_reference,
    metrics,
)
from casmi_ml.training import cached, configure, load_checkpoint, probabilities


def audit(root, coconut):
    root = Path(root)
    marker = root / 'known_spectrum_report.json'
    if marker.exists():
        return json.loads(marker.read_text())
    configure()
    selection = json.loads((root / 'selection.json').read_text())
    frame = pd.read_parquet(root / 'diagnostic.parquet')
    catalog = pd.read_parquet(root / 'catalog.parquet')
    reference_dir = root / 'reference' / 'known_diagnostic'
    if not (reference_dir / 'complete.json').exists():
        source = json.loads((root / 'manifest.json').read_text())['source']
        build_reference(source, catalog, frame, reference_dir, final=True,
                        excluded_source='enveda-np-examples')
    reference = ReferenceIndex(reference_dir)
    # Only structures actually observed in the reference spectra may be added.
    # Query labels do not enter through the full catalog.
    reference_catalog = catalog[catalog.inchikey14.isin(set(reference.rows.inchikey14))]
    pool = build_candidates(reference_catalog, coconut, final=True)
    evaluation = Evaluation(frame, pool, reference)
    rankings = []
    for path in selection['checkpoints']:
        model, checkpoint = load_checkpoint(path)
        dataset = SpectrumDataset(cached(root, 'diagnostic', checkpoint['preprocessing']))
        rankings.append(evaluation.rank(probabilities(model, dataset)))
    selected = fused(evaluation, rankings, selection['neural_weight'])
    report, per_molecule = metrics(selected, evaluation.candidates)
    baseline, baseline_rows = metrics(evaluation.baseline, evaluation.candidates)
    report['baseline'] = baseline
    report['vs_baseline'] = bootstrap_difference(per_molecule, baseline_rows)
    report['protocol'] = 'NP-source queries; exclude that entire source from reference; neural training excludes all diagnostic molecules; candidate structures from observed reference spectra and COCONUT only.'
    report['acceptance_tolerance'] = .005
    report['accepted'] = report['mrr25'] >= baseline['mrr25'] - .005
    per_molecule.to_csv(root / 'known_spectrum_per_molecule.csv', index=False)
    write_json(marker, report)
    print(json.dumps(report, indent=2), flush=True)
    return report
