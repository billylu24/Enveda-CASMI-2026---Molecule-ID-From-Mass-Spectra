import unittest

from casmi_ml.energy_conditioned_precision_audit import bootstrap_summary


class MolecularPrecisionTest(unittest.TestCase):
    def test_ratio_keeps_molecule_clusters_and_fixed_seed(self):
        one = bootstrap_summary([1, 3], [1, 3], repetitions=200)
        self.assertEqual(one["paired_difference"], 1.0)
        self.assertEqual(one["paired_molecule_bootstrap95"], [1.0, 1.0])
        self.assertEqual(one, bootstrap_summary([1, 3], [1, 3], repetitions=200))

    def test_invalid_or_empty_denominators_rejected(self):
        for n, d in [([], []), ([1], [0]), ([1, 2], [1]), ([1], [-1])]:
            with self.assertRaises(ValueError):
                bootstrap_summary(n, d, repetitions=100)
