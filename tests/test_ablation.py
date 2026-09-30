"""Check chemical mass conversion and routing invariants on independent examples."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

from baseline import PROTON, formula_mass
from casmi_ml.ablation import extended_masses, route
from casmi_ml.data import fit_preprocessing
from casmi_ml.models import FingerprintModel
from casmi_ml.secondary_inference import low_confidence_rank, predict


class AblationTests(unittest.TestCase):
    def test_deployment_fusion_matches_frozen_experiment(self):
        record = {'rankings': {'coconut15': ['a', 'b', 'c']}, 'current_full': ['a', 'd', 'b'],
                  'neural': ['d', 'c', 'b', 'a'], 'confidence': .2, 'margin': .01}
        config = {'kind': 'free_top1', 'base': 'coconut15', 'threshold': .5, 'weight': .75}
        self.assertEqual(route(record, config), low_confidence_rank(
            record['rankings']['coconut15'], record['current_full'], record['neural'], .75))

    def test_offline_deployment_exercises_both_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mass = formula_mass('C3H8')
            key = Chem.MolToInchiKey(Chem.MolFromSmiles('CCC'))[:14]
            pd.DataFrame([{'inchikey14': key, 'normalized_smiles': 'CCC', 'molecular_formula': 'C3H8',
                           'ms2_mzs': [42.], 'ms2_normalized_intensities': [1.], 'precursor_error_ppm': 0.}]).to_parquet(root/'train.parquet')
            rows = [{'molecule_id': name, 'precursor_mz': mass+PROTON, 'adduct': '[M+H]+',
                     'ms2_mzs': peaks, 'ms2_normalized_intensities': intensity, 'collision_energy_ev': [],
                     'ionization_mode': 'positive', 'instrument_type': 'test'}
                    for name, peaks, intensity in [('known', [42.], [1.]), ('unknown', [], [])]]
            frame = pd.DataFrame(rows)
            frame.to_parquet(root/'test.parquet')
            pd.DataFrame([{'inchikey': key, 'canonical_smiles': 'CCC', 'exact_mass': mass}]).to_parquet(root/'coconut.parquet')
            prep = fit_preprocessing(frame)
            dim = 6 + sum(len(v)+1 for v in prep['categories'].values())
            model = FingerprintModel('metadata', dim)
            torch.save({'architecture': 'metadata', 'metadata_dim': dim, 'preprocessing': prep,
                        'state_dict': model.state_dict()}, root/'model.pt')
            recipe = {'config': {'kind': 'free_top1', 'base': 'coconut15', 'threshold': .5, 'weight': .75},
                      'checkpoint': 'model.pt', 'checkpoint_sha256': hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()}
            (root/'recipe.json').write_text(json.dumps(recipe))
            result = predict(root/'recipe.json', root, root/'coconut.parquet', root/'submission.csv')
            self.assertEqual(result.smiles.tolist(), ['CCC', 'CCC'])
            report = json.loads((root/'submission.csv.report.json').read_text())
            self.assertEqual(report['protected_molecules'], 1)
            self.assertEqual(report['neural_molecules'], 1)

            import joblib
            import numpy as np
            from sklearn.dummy import DummyClassifier

            from casmi_ml.router_features import FEATURES
            for constant in [0, 1]:
                router = DummyClassifier(strategy='constant', constant=constant)
                router.fit(np.zeros((2, len(FEATURES))), [0, 1])
                joblib.dump(router, root/'router.joblib')
                recipe['router'] = {'path': 'router.joblib',
                                    'sha256': hashlib.sha256((root/'router.joblib').read_bytes()).hexdigest(),
                                    'high': 'H', 'low': 'F', 'threshold': .2}
                (root/'recipe.json').write_text(json.dumps(recipe))
                predict(root/'recipe.json', root, root/'coconut.parquet', root/'routed.csv')
                report = json.loads((root/'routed.csv.report.json').read_text())
                self.assertEqual(report['protected_molecules'], 2 * constant)
            recipe.pop('router')
            from casmi_ml.scale_models import ScaleModel
            bigger = ScaleModel('wide_enhanced', dim)
            torch.save({'architecture': 'wide_enhanced', 'metadata_dim': dim, 'preprocessing': prep,
                        'state_dict': bigger.state_dict()}, root/'model.pt')
            recipe['encoder_family'] = 'scale'
            recipe['checkpoint_sha256'] = hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()
            (root/'recipe.json').write_text(json.dumps(recipe))
            result = predict(root/'recipe.json', root, root/'coconut.parquet', root/'large_submission.csv')
            self.assertEqual(result.smiles.tolist(), ['CCC', 'CCC'])
            report = json.loads((root/'large_submission.csv.report.json').read_text())
            self.assertEqual(report['neural_molecules'], 1)
            from casmi_ml.direct_models import DirectRanker
            for architecture in ['fingerprint', 'graph']:
                ranker = DirectRanker(architecture)
                torch.save({'architecture': architecture, 'state_dict': ranker.state_dict(),
                            'encoder_sha256': recipe['checkpoint_sha256']}, root/'direct.pt')
                recipe['direct_ranker'] = {'path': 'direct.pt', 'weight': .5,
                                          'sha256': hashlib.sha256((root/'direct.pt').read_bytes()).hexdigest()}
                (root/'recipe.json').write_text(json.dumps(recipe))
                ranked = predict(root/'recipe.json', root, root/'coconut.parquet', root/'direct.csv')
                self.assertEqual(ranked.smiles.tolist(), ['CCC', 'CCC'])

    def test_charge_aware_neutral_mass(self):
        mass = formula_mass('C20H30O5')
        rows = pd.DataFrame([
            {'adduct': '[M+2H]2+', 'precursor_mz': (mass+2*PROTON)/2},
            {'adduct': '[M-2H]2-', 'precursor_mz': (mass-2*PROTON)/2},
            {'adduct': '[M+C2H4O2-H]-', 'precursor_mz': mass+formula_mass('C2H4O2')-PROTON},
            {'adduct': '[M]+', 'precursor_mz': mass-.00054858},
            {'adduct': 'unknown', 'precursor_mz': mass},
        ])
        result = extended_masses(rows)
        self.assertEqual(len(result), 4)
        for actual in result:
            self.assertAlmostEqual(actual, mass, places=8)

    def test_high_confidence_keeps_entire_historical_list(self):
        record = {'rankings': {'coconut15': ['a', 'b', 'c']}, 'current_full': ['a', 'd', 'e'],
                  'neural': ['e', 'd', 'a'], 'confidence': .9, 'margin': .4}
        config = {'kind': 'free_top1', 'base': 'coconut15', 'threshold': .5, 'weight': .75}
        self.assertEqual(route(record, config), ['a', 'b', 'c'])
        record['confidence'] = .2
        record['neural'] = ['e', 'd', 'c', 'b', 'a']
        self.assertNotEqual(route(record, config)[0], 'a')
        config['kind'] = 'protect_top1'
        ranking = route(record, config)
        self.assertEqual(ranking[0], 'a')
        self.assertEqual(len(ranking), len(set(ranking)))

    def test_margin_gate_preserves_ambiguous_threshold_exception(self):
        record = {'rankings': {'coconut15': ['a', 'b']}, 'confidence': .3, 'margin': .2}
        config = {'kind': 'margin', 'base': 'coconut15', 'threshold': .5, 'margin': .15, 'weight': .75}
        self.assertEqual(route(record, config), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
