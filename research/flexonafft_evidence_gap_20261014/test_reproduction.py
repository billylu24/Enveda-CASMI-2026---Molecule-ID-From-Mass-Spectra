"""Focused tests of upstream median evidence, duplicate rejection and slot integrity.

Load definitions with AST only; never execute upstream inference during unit tests.
"""
import ast
import json
import math
import time
import types
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
NB = json.loads((ROOT / 'source/casmi26-closing-the-evidence-gap.ipynb').read_text())
CODE = [''.join(c['source']) for c in NB['cells'] if c['cell_type'] == 'code']
SCOPE = dict(np=np, math=math, time=time)
TREE = ast.parse(CODE[8])
exec(compile(ast.Module(body=[x for x in TREE.body if isinstance(x, ast.ClassDef)], type_ignores=[]), '<original-iteration07>', 'exec'), SCOPE)
Fallback = SCOPE['ReplicatedFragmentFallback']

class FakeFragments:
    """Two candidates each explain a different peak; no chemistry simplification in release."""
    def mol_graph(self, smile): return (None, [0], None)
    def resolve_cfg(self, cfg): return dict(w1=1., w2=.6, tol=.005)
    def candidate_levels(self, smile, cfg, cache):
        return [np.array([10. if smile == 'a' else 20.]), np.array([]), np.array([])]
    def _prep_spectrum(self, spectrum, cfg):
        return (np.array(spectrum), np.array([10., 20.]), 1, 2)

def scorer():
    f = Fallback(types.SimpleNamespace(scope={'frag_rescore': FakeFragments()}))
    f.cache = {}; f.errors = []
    return f

class EvidenceTests(unittest.TestCase):
    def test_release_keeps_every_original_code_cell(self):
        nb = json.loads((ROOT / 'release/reproduction.ipynb').read_text())
        cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(cells[:-1], CODE)
        metadata = json.loads((ROOT / 'release/kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private'])
        self.assertFalse(metadata['enable_internet'])
        self.assertEqual(len(metadata['dataset_sources']), 15)
        for ref in metadata['dataset_sources']: self.assertEqual(len(ref.split('/')), 3)

    def test_median_requires_three_distinct_informative_spectra(self):
        f = scorer(); deadline = time.monotonic()+20
        self.assertEqual(f.score([[.8,.2],[.1,.9]], ['a','b'], deadline), {})
        self.assertEqual(f.score([[.8,.2],[.1,.9],[.1,.9]], ['a','b'], deadline), {})
        scores = f.score([[.8,.2],[.1,.9],[.2,.8]], ['a','b'], deadline)
        self.assertEqual(scores, {'a': .2, 'b': .8})
        self.assertEqual(f.errors, [])

    def test_shared_evidence_and_weak_intensity_cannot_admit_group(self):
        self.assertIsNone(Fallback.contrast(np.ones((2,3)), np.array([.2,.3,.5])))
        self.assertIsNone(Fallback.contrast(np.array([[1.,0.],[0.,0.]]), np.array([.5,.5])))
        credits = np.array([[1.,0.,0.],[0.,1.,0.]])
        self.assertIsNone(Fallback.contrast(credits, np.array([.01,.01,.98])))

    def test_expired_budget_and_bond_cap_leave_group_unchanged(self):
        f = scorer()
        self.assertEqual(f.score([[.8,.2],[.1,.9],[.2,.8]], ['a','b'], time.monotonic()-1), {})
        f.scope['frag_rescore'].mol_graph = lambda s: (None, range(41), None)
        self.assertEqual(f.score([[.8,.2],[.1,.9],[.2,.8]], ['a','b'], time.monotonic()+20), {})

    def test_only_eligible_positions_may_move(self):
        f = scorer(); f.preserve_slots([0,2,1,3], {1,2})
        with self.assertRaises(AssertionError): f.preserve_slots([1,0,2,3], {1,2})
        with self.assertRaises(AssertionError): f.preserve_slots([0,1,1,3], {1,2})

if __name__ == '__main__': unittest.main()
