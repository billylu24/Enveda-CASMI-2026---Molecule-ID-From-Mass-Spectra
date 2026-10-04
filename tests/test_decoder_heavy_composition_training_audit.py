import unittest

import torch

from casmi_ml.decoder_heavy_composition_training_audit import heavy_composition_loss


class HeavyCompositionLossTests(unittest.TestCase):
    def test_expected_mixture_count_and_padding_gradient(self):
        logits = torch.zeros((1, 3, 3), requires_grad=True)
        labels = torch.tensor([[1, 2, 0]])
        mapping = torch.zeros((3, 14))
        mapping[1, 0] = 1
        mapping[2, 2] = 1
        target = torch.zeros((1, 14))
        target[0, 0] = 1
        target[0, 2] = 1
        long = heavy_composition_loss(logits, labels, mapping, target)
        short = heavy_composition_loss(logits[:, :2], labels[:, :2], mapping, target)
        self.assertTrue(torch.equal(long, short))
        # Two uniform positions predict2/3 of each atom versus one true atom;
        # dividing each by totalheavy2 yields residual1/6 per element.
        self.assertAlmostEqual(float(long.detach()), 2 * (1 / 6 - 0.005), places=6)
        long.sum().backward()
        self.assertTrue(
            torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))
        )
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0, 0, 1]), 0.0)

    def test_per_molecule_length_normalization(self):
        logits = torch.zeros((2, 4, 2))
        labels = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])
        mapping = torch.zeros((2, 14))
        mapping[1, 0] = 1
        targets = torch.zeros((2, 14))
        targets[:, 0] = torch.tensor([1, 4])
        loss = heavy_composition_loss(logits, labels, mapping, targets)
        self.assertEqual(float(loss[0]), float(loss[1]))

    def test_hydrogen_and_empty_targets_rejected(self):
        logits = torch.zeros((1, 1, 2))
        labels = torch.ones((1, 1), dtype=torch.long)
        mapping = torch.zeros((2, 14))
        for targets in [torch.zeros((1, 14)), torch.ones((1, 14))]:
            with self.assertRaises(ValueError):
                heavy_composition_loss(logits, labels, mapping, targets)
