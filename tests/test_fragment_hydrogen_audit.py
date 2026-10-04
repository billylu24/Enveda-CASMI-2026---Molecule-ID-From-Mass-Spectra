import unittest

from casmi_ml.fragment_hydrogen_audit import component_hydrogen_offsets


class FragmentHydrogenTests(unittest.TestCase):
    def test_cut_preserves_original_hydrogen_and_isotope_counts(self):
        self.assertIn(0, component_hydrogen_offsets("CCc1ccccc1")[1])
        self.assertEqual(component_hydrogen_offsets("c1ccccc1")[0], ())
        self.assertIn(0, component_hydrogen_offsets("CC(=O)c1ccccc1")[2])

    def test_does_not_treat_whole_molecule_as_cut_component(self):
        self.assertEqual(component_hydrogen_offsets("Cc1ccccc1")[1], ())
        with self.assertRaises(ValueError):
            component_hydrogen_offsets("CC.[Na+]")


if __name__ == "__main__":
    unittest.main()
