import unittest

import numpy as np

from casmi_ml.aromatic_h_capacity_audit import CHANNELS, refine_interactions


class AromaticHCapacityTests(unittest.TestCase):
    def test_only_three_aromatic_ion_channels_are_refined(self):
        original = np.ones((2, 16, 60), np.float32)
        support = np.zeros((2, 16, 6), bool)
        refined = refine_interactions(original, support)
        keep = np.ones(60, bool)
        keep[CHANNELS] = False
        np.testing.assert_array_equal(refined[..., keep], original[..., keep])
        self.assertFalse(refined[..., CHANNELS].any())
        np.testing.assert_array_equal(
            refine_interactions(original, np.ones_like(support)), original
        )
        self.assertTrue(original.all())
        with self.assertRaises(ValueError):
            refine_interactions(original, np.ones((2, 16, 6)) * 0.1)


if __name__ == "__main__":
    unittest.main()
