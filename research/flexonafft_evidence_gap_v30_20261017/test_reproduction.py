"""Verify source identity and controlled BIO-union/rank2 behavior without inference."""
import ast
import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from prepare_release import ROOT,code

NB=json.loads((ROOT/'source/casmi26-closing-the-evidence-gap.ipynb').read_text())
TREE=ast.parse(''.join(NB['cells'][4]['source']))
SOURCES=ast.literal_eval(next(n.value for n in TREE.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_CELLS' for t in n.targets)))
FILES=ast.literal_eval(next(n.value for n in ast.parse(SOURCES[13]).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENG_FILES' for t in n.targets)))
SCOPE={};exec(compile(FILES['bio_union_extension.py'],'<upstream-bio-union>','exec'),SCOPE)
RANK={};exec(compile(ast.Module(body=[n for n in ast.parse(SOURCES[36]).body if isinstance(n,ast.FunctionDef) and n.name in ('e2r2_make_key','e2r2_row')],type_ignores=[]),'<upstream-rank2>','exec'),RANK)

class ReproductionTests(unittest.TestCase):
    def test_original_code_and_embedded_mirrors_are_exact(self):
        release=json.loads((ROOT/'release/reproduction.ipynb').read_text())
        self.assertEqual(code(release)[:-1],code(NB))
        self.assertEqual(len(code(NB)),6)
        for index,source in SOURCES.items():
            self.assertEqual((ROOT/f'source/embedded/cell_{index:02}.py').read_text(),source)
            ast.parse(source)
        meta=json.loads((ROOT/'release/kernel-metadata.json').read_text())
        self.assertFalse(meta['enable_internet']);self.assertTrue(meta['is_private'])
        self.assertEqual(len(meta['dataset_sources']),16)
        self.assertTrue(all(len(x.split('/'))==3 for x in meta['dataset_sources']))

    def union(self,root,fp=None):
        new=root/'new';new.mkdir()
        (new/'bio_meta.pkl').write_bytes(pickle.dumps(dict(keys=['A'],smiles=['preferred'])))
        np.save(new/'bio_fp.npy',np.array([[1,0]],dtype=np.uint8));np.save(new/'bio_mass.npy',np.array([100.]))
        old=root/'old.pkl';old.write_bytes(pickle.dumps(dict(keys=['A','legacyB','legacyB2','bad'],smiles=['older','B','B','invalid'])))
        calls=[]
        def fingerprint(s):calls.append(s);return np.array([0,1],dtype=np.uint8),120.
        with contextlib.redirect_stdout(io.StringIO()):
            result=SCOPE['BioUnion'].load(new,old,fp or fingerprint,lambda s:None if s=='invalid' else s,root/'report.json')
        return result,json.loads((root/'report.json').read_text()),calls

    def test_union_preserves_new_priority_and_recalculates_unique_additions(self):
        with tempfile.TemporaryDirectory() as d:
            (meta,fp,mass),report,calls=self.union(Path(d))
        self.assertEqual(meta,dict(keys=['A','B'],smiles=['preferred','B']))
        np.testing.assert_array_equal(fp,[[1,0],[0,1]]);np.testing.assert_array_equal(mass,[100.,120.])
        self.assertEqual(calls,['B'])
        self.assertEqual((report['added_rows'],report['overlap_rows'],report['invalid_rows']),(1,2,1))

    def test_incompatible_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(AssertionError,'Incompatible'):
            self.union(Path(d),lambda s:(np.array([1,0],dtype=np.float32),120.))

    def test_deadline_raises_instead_of_accepting_partial_pool(self):
        with tempfile.TemporaryDirectory() as d,patch.object(SCOPE['time'],'monotonic',side_effect=[0,1201]),self.assertRaises(TimeoutError):
            self.union(Path(d))

    def test_rank2_respects_metric_twins_top1_and_remaining_order(self):
        key=RANK['e2r2_make_key'](lambda s:None if s=='invalid' else str(s).lower())
        rank=RANK['e2r2_row']
        self.assertEqual(rank(['A','b','C'],['a'],key),(['A','b','C'],'same_as_rank1',None))
        self.assertEqual(rank(['A','x','B','b','C'],['b'],key),(['A','b','x','C'],'inserted','3'))
        self.assertEqual(rank(['A','x','B'],['invalid','b'],key),(['A','x','B'],'invalid_x2',None))

if __name__=='__main__':unittest.main()
