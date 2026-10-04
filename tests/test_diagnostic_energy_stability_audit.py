import unittest

import numpy as np

from casmi_ml.diagnostic_energy_stability_audit import pair_kind, stability_summary


class DiagnosticEnergyStabilityTest(unittest.TestCase):
    def test_merged_energies_missing_and_duplicates_are_not_singleton_pairs(self):
        self.assertEqual(pair_kind([20, 40], [30]), "multi_energy_signature")
        self.assertEqual(pair_kind(None, [30]), "missing_energy")
        self.assertEqual(pair_kind([20], [40], same_row=True), "duplicated_single_row")
        self.assertEqual(
            pair_kind([20], [40], same_instrument=False), "different_instrument"
        )
        self.assertEqual(pair_kind([20, 20], [20]), "same_single_energy")
        self.assertEqual(pair_kind([20], [40]), "different_single_energy")

    def test_empty_signatures_do_not_inflate_jaccard_and_direction_is_preserved(self):
        left, right = np.zeros((3, 60)), np.zeros((3, 60))
        left[1, 0] = 0.5
        right[1, 1] = 0.75
        left[2, 2] = 0.25
        right[2, 2] = 0.5
        result = stability_summary(left, right, [True, False, True])
        self.assertEqual(result["mean_jaccard_on_nonempty_pairs"], 0.5)
        self.assertEqual(result["pairs_with_evidence_in_either_row"], 2)
        self.assertEqual(result["channels_gained_at_higher_energy"][:3], [1, 0, 0])
        self.assertEqual(result["channels_lost_at_higher_energy"][:3], [0, 1, 0])
        self.assertEqual(
            result["channel_mean_higher_minus_lower_strength"][2], 0.25 / 3
        )
        self.assertIsNone(
            stability_summary(left[:1], right[:1])["mean_jaccard_on_nonempty_pairs"]
        )

    def test_invalid_shape_or_nonfinite_observations_rejected(self):
        with self.assertRaises(ValueError):
            stability_summary(np.zeros((2, 59)), np.zeros((2, 59)))
        x = np.zeros((1, 60))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            stability_summary(x, np.zeros((1, 60)))
