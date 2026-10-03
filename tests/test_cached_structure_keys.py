import unittest

from rdkit import Chem

from casmi_ml.cached_structure_keys import structure_key


class CachedStructureKeyTests(unittest.TestCase):
    def test_actual_smiles_identity_and_stereochemical_connectivity_are_preserved(self):
        structure_key.cache_clear()
        for smiles in [
            "CCO",
            "OCC",
            "C[C@H](O)F",
            "C[C@@H](O)F",
            "[Na+].[O-]C(=O)C",
            "[13CH3]CO",
        ]:
            self.assertEqual(
                structure_key(smiles),
                Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14],
            )
            self.assertEqual(
                structure_key(smiles),
                Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14],
            )
        self.assertGreaterEqual(structure_key.cache_info().hits, 6)
        self.assertEqual(structure_key.cache_info().maxsize, 100000)
        with self.assertRaisesRegex(ValueError, "Invalid current"):
            structure_key("invalid-smiles")
