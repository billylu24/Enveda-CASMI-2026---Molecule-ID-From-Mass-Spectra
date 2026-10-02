import unittest

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from casmi_ml.generation_augmentation import enumerated_target
from casmi_ml.research_models import SmilesVocabulary


class GenerationAugmentationTests(unittest.TestCase):
    def test_deterministic_epoch_enumeration_preserves_chemical_identity(self):
        smiles = "COc1ccc(CC(O)C(=O)O)cc1"
        vocabulary = SmilesVocabulary.fit([smiles])
        outcomes = []
        for epoch in range(1, 7):
            tokens, stats = enumerated_target(smiles, "molecule", epoch, vocabulary)
            self.assertEqual(
                (tokens, stats),
                enumerated_target(smiles, "molecule", epoch, vocabulary),
            )
            decoded = vocabulary.decode(tokens)
            a, b = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(decoded)
            self.assertEqual(Chem.MolToInchiKey(a), Chem.MolToInchiKey(b))
            self.assertEqual(
                rdMolDescriptors.CalcMolFormula(a), rdMolDescriptors.CalcMolFormula(b)
            )
            self.assertAlmostEqual(Descriptors.ExactMolWt(a), Descriptors.ExactMolWt(b))
            outcomes.append(decoded)
        self.assertGreater(len(set(outcomes)), 1)

    def test_unencodable_augmentation_keeps_original_target(self):
        class OnlyOriginalVocabulary:
            def encode(self, smiles):
                return [1, 5, 2] if smiles == "CCO" else None

        from unittest.mock import patch

        with patch(
            "casmi_ml.generation_augmentation.Chem.MolToSmiles",
            return_value="OCC",
        ):
            sequence, stats = enumerated_target(
                "CCO", "molecule", 1, OnlyOriginalVocabulary()
            )
        self.assertEqual(sequence, [1, 5, 2])
        self.assertTrue(stats["vocabulary_or_length_fallback"])
        self.assertFalse(stats["changed"])


if __name__ == "__main__":
    unittest.main()
