import unittest

import torch

from casmi_ml.decoder_interpolation import interpolate_decoder


class DecoderInterpolationTests(unittest.TestCase):
    def test_endpoints_midpoint_and_unchanged_integer_buffers(self):
        a = {"weight": torch.tensor([1.0, 3.0]), "buffer": torch.tensor([1])}
        b = {"weight": torch.tensor([3.0, 5.0]), "buffer": torch.tensor([1])}
        for alpha, expected in [
            (0, a["weight"]),
            (0.5, torch.tensor([2.0, 4.0])),
            (1, b["weight"]),
        ]:
            c = interpolate_decoder(a, b, alpha)
            self.assertTrue(torch.equal(c["weight"], expected))
            self.assertTrue(torch.equal(c["buffer"], a["buffer"]))
        self.assertTrue(torch.equal(a["weight"], torch.tensor([1.0, 3.0])))

    def test_incompatible_or_nonfinite_checkpoints_are_rejected(self):
        a = {"weight": torch.tensor([1.0])}
        for b in [
            {"other": torch.tensor([1.0])},
            {"weight": torch.tensor([1.0, 2.0])},
            {"weight": torch.tensor([float("nan")])},
        ]:
            with self.assertRaises(ValueError):
                interpolate_decoder(a, b, 0.5)
        with self.assertRaises(ValueError):
            interpolate_decoder(
                {"buffer": torch.tensor([1])}, {"buffer": torch.tensor([2])}, 0.5
            )
        for alpha in [-1, 2, float("nan")]:
            with self.assertRaises(ValueError):
                interpolate_decoder(a, a, alpha)


if __name__ == "__main__":
    unittest.main()
