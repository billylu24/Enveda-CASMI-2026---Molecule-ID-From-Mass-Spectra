import unittest

import torch

from casmi_ml.generator_composition import compose


class GeneratorCompositionTests(unittest.TestCase):
    def test_only_formula_head_is_replaced_and_inputs_are_preserved(self):
        a = {
            "config": {"encoder_sha256": "encoder"},
            "vocabulary": ["C"],
            "condition_dim": 2,
            "formula": {"weight": torch.tensor([1.0])},
            "decoder": {"weight": torch.tensor([3.0])},
        }
        b = {**a, "formula": {"weight": torch.tensor([5.0])}}
        c = compose(a, b)
        self.assertTrue(torch.equal(c["decoder"]["weight"], a["decoder"]["weight"]))
        self.assertTrue(torch.equal(c["formula"]["weight"], b["formula"]["weight"]))
        c["formula"]["weight"][0] = 99
        self.assertEqual(a["formula"]["weight"][0], 1)
        self.assertEqual(b["formula"]["weight"][0], 5)

    def test_incompatible_conditioner_and_formula_are_rejected(self):
        a = {
            "config": {"encoder_sha256": "encoder"},
            "vocabulary": ["C"],
            "condition_dim": 2,
            "formula": {"weight": torch.tensor([1.0])},
        }
        for delta in [
            {"config": {"encoder_sha256": "other"}},
            {"vocabulary": ["O"]},
            {"condition_dim": 3},
            {"formula": {"weight": torch.tensor([float("nan")])}},
            {"formula": {"weight": torch.tensor([1.0, 2.0])}},
        ]:
            with self.assertRaises(ValueError):
                compose(a, {**a, **delta})


if __name__ == "__main__":
    unittest.main()
