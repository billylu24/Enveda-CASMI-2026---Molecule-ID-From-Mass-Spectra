import unittest

import torch

from casmi_ml.generation_experiment import formula_counts
from casmi_ml.mass_formula_generation_pilot import mixed_condition


class FormulaPilotConditionTest(unittest.TestCase):
    def test_fixed_budget_preserves_first_half_and_assigns_proposals_evenly(self):
        z = torch.tensor([[1.0, 2.0]])
        soft = torch.arange(14, dtype=torch.float32)[None] / 32
        condition = mixed_condition(z, soft, ["CH4", "C2H6"])
        self.assertEqual(condition.shape, (128, 16))
        self.assertTrue(
            torch.equal(condition[:64], torch.cat([z, soft], 1).expand(64, -1))
        )
        self.assertTrue(torch.equal(condition[:, :2], z.expand(128, -1)))
        self.assertTrue(
            torch.equal(
                condition[64, 2:],
                torch.tensor(formula_counts("CH4"), dtype=torch.float32) / 32,
            )
        )
        self.assertTrue(
            torch.equal(
                condition[65, 2:],
                torch.tensor(formula_counts("C2H6"), dtype=torch.float32) / 32,
            )
        )
        self.assertTrue(torch.equal(condition[66], condition[64]))

    def test_missing_proposals_retains_all_original_conditions(self):
        z = torch.ones(1, 3)
        soft = torch.zeros(1, 14)
        self.assertTrue(
            torch.equal(
                mixed_condition(z, soft, []), torch.cat([z, soft], 1).expand(128, -1)
            )
        )
        with self.assertRaises(ValueError):
            mixed_condition(z, soft, [], samples=256)
