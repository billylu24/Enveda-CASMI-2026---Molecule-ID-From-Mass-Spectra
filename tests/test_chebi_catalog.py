import unittest

from rdkit import Chem

from casmi_ml.chebi_catalog import normalize


class ChebiImportTests(unittest.TestCase):
    def row(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return {
            "compound_id": "42",
            "status_id": "1",
            "default_structure": "true",
            "smiles": smiles,
            "standard_inchi_key": Chem.MolToInchiKey(mol),
        }

    def test_connectivity_is_preserved_and_stereo_deduplicates(self):
        first, why = normalize(self.row("C[C@H](O)C(=O)O"))
        second, _ = normalize(self.row("C[C@@H](O)C(=O)O"))
        self.assertEqual(why, "retained")
        self.assertEqual(first, second)
        self.assertNotIn("@", first["normalized_smiles"])
        self.assertAlmostEqual(first["mass"], 90.031694, places=5)

    def test_invalid_scope_and_identity_are_excluded(self):
        for smiles in ["CCO.[Na+]", "C[N+](C)(C)C", "[13CH3]CO", "*CO", "O"]:
            self.assertIsNone(normalize(self.row(smiles))[0])
        row = self.row("CCO")
        row["standard_inchi_key"] = "X" * 27
        self.assertEqual(normalize(row)[1], "identity_disagreement")
        row = self.row("CCO")
        row["status_id"] = "9"
        self.assertEqual(normalize(row)[1], "not_reviewed_default")
