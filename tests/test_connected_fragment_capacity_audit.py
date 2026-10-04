import unittest

import numpy as np

from casmi_ml.connected_fragment_capacity_audit import ION_CHANNELS, refine_interactions


class ConnectedCapacityTests(unittest.TestCase):
    def test_only_observed_six_ion_channels_are_refined(self):
        original = np.ones((2, 3, 60), np.float32)
        original[0, 0, 24] = 0
        connected = np.ones((2, 3, 12), bool)
        connected[0, 0, :6] = False
        refined = refine_interactions(original, connected)
        keep = np.ones(60, bool)
        keep[ION_CHANNELS] = False
        np.testing.assert_array_equal(refined[..., keep], original[..., keep])
        self.assertFalse(refined[0, 0, ION_CHANNELS].any())
        np.testing.assert_array_equal(original[1], refined[1])
        self.assertEqual(original[0, 0, 27], 1)

    def test_missing_chemistry_and_full_support_preserve_inputs(self):
        original = np.zeros((4, 60), np.float32)
        np.testing.assert_array_equal(
            refine_interactions(original, np.zeros((4, 12))), original
        )
        original[:] = 0.3
        np.testing.assert_array_equal(
            refine_interactions(original, np.ones((4, 12))), original
        )
        with self.assertRaises(ValueError):
            refine_interactions(original, np.ones((4, 12)) * 0.5)
        with self.assertRaises(ValueError):
            refine_interactions(original, np.ones((3, 12)))


if __name__ == "__main__":
    unittest.main()
