import unittest

import numpy as np
import torch

from casmi_ml.formula_budget_sampling import atom_count_tokens, sample
from casmi_ml.generation_experiment import ELEMENTS
from casmi_ml.research_models import SmilesDecoder, SmilesVocabulary


class FormulaBudgetSamplingTest(unittest.TestCase):
    def test_tokens_count_heavy_atoms_and_flag_unsupported_atoms(self):
        vocabulary = SmilesVocabulary(
            ["C", "c", "[nH]", "[13CH3+]", "Cl", "[Se]", "[se]", "(", ")", "1", "H"]
        )
        counts, unsupported = atom_count_tokens(vocabulary)
        self.assertEqual(counts[vocabulary.ids["[13CH3+]"], ELEMENTS.index("C")], 1)
        self.assertEqual(counts[vocabulary.ids["[nH]"], ELEMENTS.index("N")], 1)
        self.assertEqual(counts[vocabulary.ids["Cl"], ELEMENTS.index("Cl")], 1)
        self.assertFalse(counts[:, ELEMENTS.index("H")].any())
        self.assertTrue(unsupported[vocabulary.ids["[Se]"]])
        self.assertTrue(unsupported[vocabulary.ids["[se]"]])
        self.assertFalse(unsupported[vocabulary.ids["("]])

    def test_inactive_budget_keeps_original_tokens_and_likelihood_exact(self):
        torch.manual_seed(42)
        vocabulary = SmilesVocabulary(["C", "O", "N", "(", ")", "1", "="])
        decoder = SmilesDecoder(
            len(vocabulary.tokens), 3, width=8, layers=1, limit=12
        ).eval()
        condition = torch.tensor([[0.1, 0.2, 0.3]])
        counts, unsupported = atom_count_tokens(vocabulary)
        original = decoder.generate(
            condition,
            128,
            temperature=0.8,
            generator=torch.Generator().manual_seed(731),
        )
        constrained = sample(
            decoder,
            condition,
            np.zeros((128, 14)),
            np.zeros(128, bool),
            counts,
            unsupported,
            generator=torch.Generator().manual_seed(731),
        )
        for left, right in zip(original, constrained):
            self.assertTrue(torch.equal(left, right))

    def test_active_budget_limits_atoms_and_blocks_early_eos(self):
        class Decoder:
            limit = 6
            head = type("Head", (), {"out_features": 5})()

            def eval(self):
                return self

            def __call__(self, tokens, condition):
                logits = torch.full((128, len(tokens[0]), 5), -100.0)
                logits[:, -1, 2] = 10
                logits[:, -1, 4] = 9
                return logits

        counts = np.zeros((5, 14), int)
        counts[4, 0] = 1
        targets = np.zeros((128, 14), int)
        targets[:, 0] = 1
        tokens, _, finished = sample(
            Decoder(),
            torch.zeros(1, 1),
            targets,
            np.ones(128, bool),
            counts,
            np.zeros(5, bool),
            generator=torch.Generator().manual_seed(42),
        )
        self.assertTrue(finished.all())
        self.assertTrue(torch.equal(tokens, torch.tensor([1, 4, 2]).expand(128, -1)))
