import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from casmi_ml.neighborhood_diagnostic_audit import count_features


class NeighborhoodDiagnosticTest(unittest.TestCase):
    def test_actual_count_transform_and_atom_stereo_invariance(self):
        mol = Chem.MolFromSmiles("CCCCCC")
        raw = rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=2048, includeChirality=False
        ).GetCountFingerprintAsNumPy(mol)
        np.testing.assert_allclose(count_features("CCCCCC", 3), np.log1p(raw))
        for radius in (2, 3):
            np.testing.assert_array_equal(
                count_features("CCO", radius), count_features("OCC", radius)
            )
            np.testing.assert_array_equal(
                count_features("C[C@H](O)CC", radius),
                count_features("C[C@@H](O)CC", radius),
            )

    def test_radius_depth_separates_original_local_alias(self):
        left, right = "CCCCCC=CCCCCCCC", "CCCCCCC=CCCCCCC"
        np.testing.assert_array_equal(count_features(left, 2), count_features(right, 2))
        self.assertFalse(
            np.array_equal(count_features(left, 3), count_features(right, 3))
        )
        for smiles, radius in [("invalid", 2), ("", 3), ("CC", 4)]:
            with self.assertRaises(ValueError):
                count_features(smiles, radius)
