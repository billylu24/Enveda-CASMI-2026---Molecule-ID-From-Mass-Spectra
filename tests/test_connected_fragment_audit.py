import unittest

import numpy as np

from casmi_ml.connected_fragment_audit import connected_support, pair_summary


class ConnectedFragmentTests(unittest.TestCase):
    def test_methoxy_composition_requires_actual_detachable_component(self):
        self.assertTrue(connected_support("COc1ccccc1")[6])
        self.assertFalse(connected_support("Oc1ccccc1")[6])
        self.assertFalse(connected_support("CO")[6])

    def test_cyclic_bonds_and_disconnected_salts_do_not_create_fragments(self):
        self.assertFalse(any(connected_support("c1ccccc1")))
        self.assertFalse(any(connected_support("COc1ccccc1.[Na+]")))
        with self.assertRaises(ValueError):
            connected_support("invalid")

    def test_observed_ties_include_both_directions_and_no_absent_signal(self):
        observed = np.zeros((2, 36))
        observed[:, 18] = 1
        broad = np.zeros((2, 12), bool)
        broad[:, 6] = True
        specific = broad.copy()
        specific[1, 6] = False
        result = pair_summary(
            observed, broad, specific, np.array([[1], [0]]), np.ones((2, 1), bool)
        )
        channel = result["channels"][18]
        self.assertEqual(channel["broad_ties_separated"], 2)
        self.assertEqual(channel["separated_ties_positive_advantage"], 1)
        self.assertEqual(channel["separated_ties_negative_advantage"], 1)
        self.assertEqual(result["channels"][19]["observed_compatible_pairs"], 0)
        self.assertEqual(result["aliases_separated_by_connected_refinement"], 2)


if __name__ == "__main__":
    unittest.main()
