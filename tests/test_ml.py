import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import features, fit_preprocessing, split_key
from casmi_ml.inference import validate_submission
from casmi_ml.models import FingerprintModel
from casmi_ml.ranking import CandidateIndex, metrics, neural_rank, rrf
from casmi_ml.training import load_checkpoint


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.row = {'ms2_mzs': [50., 100., 150.], 'ms2_normalized_intensities': [.2, 1., .5],
                    'precursor_mz': 200., 'collision_energy_ev': [10., 20.],
                    'adduct': '[M+H]+', 'ionization_mode': 'positive', 'instrument_type': 'test'}
        self.prep = fit_preprocessing(pd.DataFrame([self.row]))

    def batch(self, row=None):
        values = features(row or self.row, self.prep)
        return {k: torch.from_numpy(v[None]) for k, v in zip(['hist', 'loss', 'meta', 'peaks', 'mask'], values)}

    def test_group_split_and_diagnostic_exclusion(self):
        keys = [f'{i:014d}' for i in range(1000)]
        groups = {name: {k for k in keys if split_key(k) == name} for name in ['train', 'dev', 'holdout']}
        self.assertFalse(groups['train'] & groups['dev'])
        self.assertFalse(groups['train'] & groups['holdout'])
        self.assertEqual(split_key(keys[0], {keys[0]}), 'diagnostic')
        self.assertEqual(split_key(keys[20]), split_key(keys[20]))

    def test_unknown_missing_and_empty(self):
        row = dict(self.row, adduct='unknown', collision_energy_ev=None, precursor_mz=None,
                   ms2_mzs=[], ms2_normalized_intensities=[])
        batch = self.batch(row)
        for value in batch.values():
            self.assertTrue(torch.isfinite(value).all())
        for architecture in ['mlp', 'enhanced', 'metadata', 'deepsets', 'transformer']:
            model = FingerprintModel(architecture, batch['meta'].shape[1]).eval()
            self.assertTrue(torch.isfinite(model(batch)).all())

    def test_padding_permutation_and_roundtrip(self):
        batch = self.batch()
        for architecture in ['deepsets', 'transformer']:
            model = FingerprintModel(architecture, batch['meta'].shape[1]).eval()
            with torch.no_grad():
                before = model(batch)
                changed = {k: v.clone() for k, v in batch.items()}
                changed['peaks'][~changed['mask']] = 10.
                torch.testing.assert_close(before, model(changed), atol=1e-5, rtol=1e-5)
                perm = torch.randperm(64)
                changed['peaks'] = batch['peaks'][:, perm]
                changed['mask'] = batch['mask'][:, perm]
                torch.testing.assert_close(before, model(changed), atol=1e-5, rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                torch.save({'architecture': architecture, 'metadata_dim': batch['meta'].shape[1],
                            'state_dict': model.state_dict()}, path)
                restored, _ = load_checkpoint(path)
                torch.testing.assert_close(before, restored(batch))

    def test_mass_window_and_no_truth_injection(self):
        catalog = pd.DataFrame({'mass': [100., 100.005, 100.02], 'inchikey14': ['a', 'b', 'c'],
                                'normalized_smiles': ['CC', 'CCC', 'CCCC']})
        index = CandidateIndex(catalog)
        self.assertEqual(set(index.query(100.).inchikey14), {'a', 'b'})
        self.assertTrue(index.query(None).empty)
        self.assertNotIn('truth', index.query(100.).inchikey14.tolist())

    def test_mrr_cutoff_recall_and_fusion(self):
        rankings = {'a': ['a'], 'b': ['x', 'b'], 'c': ['x'] * 25 + ['c'], 'd': []}
        report, _ = metrics(rankings, {'a': ['a'], 'b': ['b'], 'c': ['c'], 'd': []})
        self.assertEqual(report['mrr25'], .375)
        self.assertEqual(report['candidate_recall'], .75)
        self.assertEqual(report['conditional_mrr25'], .5)
        self.assertEqual(rrf([['a', 'b'], ['b', 'a']], [0, 1]), ['b', 'a'])
        fps = np.zeros((2, 2048), dtype=np.float32)
        fps[1, 0] = 1
        p = np.full(2048, .01)
        p[0] = .99
        self.assertEqual(neural_rank(p, ['a', 'b'], fps)[0], 'b')
        self.assertEqual(neural_rank(p, ['a', 'b'], fps), neural_rank(p, ['a', 'b'], fps.astype(np.uint8)))

    def test_submission_invalid_and_duplicates(self):
        test = pd.DataFrame({'molecule_id': ['a']})
        validate_submission(test, pd.DataFrame({'molecule_id': ['a'], 'smiles': ['CC;CCC']}))
        for smiles in ['CC;C(C)', '', 'not smiles']:
            with self.assertRaises(ValueError):
                validate_submission(test, pd.DataFrame({'molecule_id': ['a'], 'smiles': [smiles]}))

    def test_real_manifest_no_leakage(self):
        root = Path('artifacts/experiment')
        if not (root / 'manifest.json').exists():
            self.skipTest('No prepared data')
        sets = {split: set(pd.read_parquet(root / f'{split}.parquet', columns=['inchikey14']).inchikey14)
                for split in ['train', 'dev', 'holdout', 'diagnostic']}
        for a, left in sets.items():
            for b, right in sets.items():
                if a != b:
                    self.assertFalse(left & right, f'{a} overlaps {b}')

class IntegrationTests(unittest.TestCase):
    def test_confidence_routing_preserves_reference_hits(self):
        from casmi_ml.guarded import guarded_rank
        baseline, neural = ['a', 'b', 'c'], ['c', 'a', 'b']
        self.assertEqual(guarded_rank(baseline, neural, .9, .65), baseline)
        self.assertEqual(guarded_rank(baseline, neural, .3, .65), ['a', 'c', 'b'])
        self.assertEqual(guarded_rank([], neural, 0., .65), neural)

    def test_known_reference_removes_identical_query_copies(self):
        from baseline import PROTON, formula_mass
        from casmi_ml.ranking import ReferenceIndex, build_reference
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mass = formula_mass('C3H8')
            frame = pd.DataFrame({'inchikey14': ['key'] * 3, 'normalized_smiles': ['CCC'] * 3,
                                  'ms2_mzs': [[42.], [42.], [43.]],
                                  'ms2_normalized_intensities': [[1.], [1.], [1.]],
                                  'precursor_error_ppm': [0.] * 3})
            frame.to_parquet(root / 'train.parquet')
            catalog = pd.DataFrame({'inchikey14': ['key'], 'mass': [mass]})
            query = pd.DataFrame({'inchikey14': ['key'], 'precursor_mz': [mass + PROTON],
                                  'adduct': ['[M+H]+'], 'ms2_mzs': [[42.]],
                                  'ms2_normalized_intensities': [[1.]]})
            build_reference(root / 'train.parquet', catalog, query, root / 'reference',
                            final=True, exclude_queries=True)
            reference = ReferenceIndex(root / 'reference')
            self.assertEqual(len(reference.rows), 1)
            self.assertIn(430, reference.matrix.indices)

    def test_deployment_gate_preserves_frozen_choice(self):
        from casmi_ml.experiment import deployment_gate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = {'checkpoints': ['model.pt'], 'architectures': ['mlp'], 'neural_weight': .5}
            frozen = json.dumps(candidate)
            (root / 'selection.json').write_text(frozen)
            (root / 'final_selection.json').write_text(frozen)
            (root / 'holdout_report.json').write_text(json.dumps({'mrr25': .01, 'baseline': {'mrr25': .02}}))
            result = deployment_gate(root)
            self.assertEqual(result['checkpoints'], [])
            self.assertEqual(result['neural_weight'], 0)
            self.assertEqual((root / 'selection.json').read_text(), frozen)
            (root / 'final_selection.json').write_text(frozen)
            (root / 'holdout_report.json').write_text(json.dumps({'mrr25': .03, 'baseline': {'mrr25': .02}}))
            self.assertEqual(deployment_gate(root)['checkpoints'], ['model.pt'])

    def test_candidate_pool_excludes_heldout_labels(self):
        from casmi_ml.ranking import build_candidates
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'coconut.parquet'
            pd.DataFrame({'inchikey': ['EXTERNAL000001-AAAA'], 'canonical_smiles': ['CCC'],
                          'exact_mass': [44.0626]}).to_parquet(path)
            catalog = pd.DataFrame({'inchikey14': ['TRAIN000000001', 'HOLDOUT0000001'],
                                    'normalized_smiles': ['CC', 'CCCC'], 'mass': [30.047, 58.078],
                                    'split': ['train', 'holdout']})
            result = build_candidates(catalog, path)
            self.assertNotIn('HOLDOUT0000001', set(result.inchikey14))
            self.assertIn('TRAIN000000001', set(result.inchikey14))
            self.assertIn('EXTERNAL000001', set(result.inchikey14))

    def test_offline_prediction_with_empty_spectrum(self):
        from rdkit import Chem

        from baseline import PROTON, formula_mass
        from casmi_ml.inference import predict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = Chem.MolToInchiKey(Chem.MolFromSmiles('CCC'))[:14]
            mass = formula_mass('C3H8')
            pd.DataFrame({'inchikey14': [key], 'normalized_smiles': ['CCC'],
                          'molecular_formula': ['C3H8'], 'ms2_mzs': [[42.]],
                          'ms2_normalized_intensities': [[1.]], 'precursor_error_ppm': [0.]}).to_parquet(root/'train.parquet')
            test = pd.DataFrame({'molecule_id': ['test'], 'precursor_mz': [mass+PROTON],
                                 'adduct': ['[M+H]+'], 'ms2_mzs': [[]], 'ms2_normalized_intensities': [[]]})
            test.to_parquet(root/'test.parquet')
            pd.DataFrame({'inchikey': [key], 'canonical_smiles': ['CCC'],
                          'exact_mass': [mass]}).to_parquet(root/'coconut.parquet')
            selection = root/'selection.json'
            selection.write_text(json.dumps({'checkpoints': [], 'architectures': [], 'neural_weight': 0.}))
            result = predict(selection, root, root/'coconut.parquet', root/'submission.csv')
            self.assertEqual(result.iloc[0].smiles, 'CCC')
            preprocessing = fit_preprocessing(pd.DataFrame([{
                'precursor_mz': mass + PROTON, 'collision_energy_ev': [],
                'adduct': '[M+H]+', 'ionization_mode': 'positive', 'instrument_type': 'test'}]))
            dim = 6 + sum(len(v) + 1 for v in preprocessing['categories'].values())
            model = FingerprintModel('mlp', dim)
            torch.save({'architecture': 'mlp', 'metadata_dim': dim, 'preprocessing': preprocessing,
                        'state_dict': model.state_dict()}, root / 'model.pt')
            selection.write_text(json.dumps({'checkpoints': ['model.pt'], 'architectures': ['mlp'], 'neural_weight': .5}))
            result = predict(selection, root, root/'coconut.parquet', root/'neural_submission.csv')
            self.assertEqual(result.iloc[0].smiles, 'CCC')


if __name__ == '__main__':
    unittest.main()
