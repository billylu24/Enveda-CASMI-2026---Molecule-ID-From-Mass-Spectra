import unittest

import torch

from casmi_ml.diagnostic_strength_transform import strong_peak_loss_features


class DiagnosticStrengthTransformTests(unittest.TestCase):
    def test_only_peak_losses_are_attenuated_and_inputs_are_unmodified(self):
        chemical = torch.linspace(0, 1, 24).repeat(3, 2, 1)
        original = chemical.clone()
        transformed = strong_peak_loss_features(chemical)
        torch.testing.assert_close(chemical, original, rtol=0, atol=0)
        torch.testing.assert_close(
            transformed[..., 0::3], chemical[..., 0::3], rtol=0, atol=0
        )
        torch.testing.assert_close(
            transformed[..., 1::3], chemical[..., 1::3], rtol=0, atol=0
        )
        self.assertTrue(bool((transformed[..., 2::3] <= chemical[..., 2::3]).all()))
        self.assertTrue(
            bool((transformed[..., 2::3][chemical[..., 2::3] <= 0.3] == 0).all())
        )
        self.assertEqual(float(transformed[..., -1].max()), 1)
        with self.assertRaisesRegex(ValueError, "24"):
            strong_peak_loss_features(torch.zeros(3, 23))
