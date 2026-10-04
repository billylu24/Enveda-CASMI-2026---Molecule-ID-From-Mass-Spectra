import unittest

import torch

from casmi_ml.decoder_mixed_formula_training_audit import (
    integer_condition_key,
    molecular_nll,
)


class DecoderMixedFormulaTrainingTest(unittest.TestCase):
    def test_padding_does_not_change_molecular_loss_or_train_padding_logits(self):
        logits = torch.tensor(
            [[[0.2, 1.0, -0.3], [0.5, 0.4, 0.3], [-100.0, 10.0, 20.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[1, 2, 0]])
        short = molecular_nll(logits[:, :2], labels[:, :2])
        full = molecular_nll(logits, labels)
        self.assertTrue(torch.equal(short, full))
        full.sum().backward()
        self.assertTrue(
            torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))
        )

    def test_molecules_keep_equal_weight_with_different_token_counts(self):
        logits = torch.zeros((2, 3, 4))
        labels = torch.tensor([[1, 0, 0], [1, 2, 3]])
        losses = molecular_nll(logits, labels)
        self.assertTrue(torch.equal(losses[0], losses[1]))
        self.assertAlmostEqual(
            float(losses.mean()), float(torch.log(torch.tensor(4.0)))
        )

    def test_condition_assignment_is_independent_of_row_order(self):
        keys = ["molecule-one", "molecule-two", "molecule-three"]
        a = {k: integer_condition_key(k) for k in keys}
        b = {k: integer_condition_key(k) for k in reversed(keys)}
        self.assertEqual(a, b)
