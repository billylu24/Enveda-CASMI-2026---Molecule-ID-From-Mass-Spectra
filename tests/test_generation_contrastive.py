import unittest
from unittest.mock import patch

import torch

from casmi_ml.generation_contrastive_pilot import prior_input
from casmi_ml.research_models import SmilesDecoder


class ContrastiveSamplingTests(unittest.TestCase):
    def test_prior_replaces_only_latent_and_preserves_query_chemistry_metadata(self):
        condition = torch.arange(10, dtype=torch.float32).reshape(1, 10)
        prior = prior_input(condition, torch.tensor([7.0, 8.0, 9.0]))
        torch.testing.assert_close(prior[0, :3], torch.tensor([7.0, 8.0, 9.0]))
        torch.testing.assert_close(prior[0, 3:], condition[0, 3:])
        torch.testing.assert_close(
            condition, torch.arange(10, dtype=torch.float32).reshape(1, 10)
        )

    def test_zero_weight_exactly_preserves_samples_and_contrast_changes_distribution(
        self,
    ):
        model = SmilesDecoder(6, 4, width=8, layers=1, limit=2)
        query = torch.ones(1, 4)
        prior = torch.zeros(1, 4)

        def forward(tokens, condition):
            logits = torch.zeros(len(tokens), tokens.shape[1], 6)
            logits[:, -1, 2] = -20
            logits[:, -1, 4] = 2 - condition[:, 0]
            logits[:, -1, 5] = condition[:, 0]
            return logits

        def sample(weight, background=None):
            return model.generate(
                query,
                samples=128,
                generator=torch.Generator().manual_seed(42),
                contrast_weight=weight,
                prior_condition=background,
            )

        with patch.object(model, "forward", side_effect=forward):
            original = sample(0)
            zero = sample(0, prior)
            contrast = sample(1, prior)
        for a, b in zip(original, zero):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater(
            (contrast[0][:, -1] == 5).sum(), (original[0][:, -1] == 5).sum()
        )
        with self.assertRaises(ValueError):
            sample(1)
        with self.assertRaises(ValueError):
            sample(float("nan"), prior)
