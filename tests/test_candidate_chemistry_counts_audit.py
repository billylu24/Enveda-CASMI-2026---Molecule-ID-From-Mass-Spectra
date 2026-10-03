import unittest

import numpy as np

from casmi_ml.candidate_chemistry_counts_audit import RULES, motif_counts
from casmi_ml.candidate_count_capacity_audit import multiplicity_features
from casmi_ml.full_gap_candidate_compatibility import structural_support


class MotifMultiplicityTests(unittest.TestCase):
    def test_multiplicity_extras_are_bounded_and_preserve_single_match_control(self):
        interactions = np.ones((1, 2, 60), dtype=np.float32)
        counts = np.ones((1, 2, 20), dtype=np.int32)
        extra = multiplicity_features(interactions, counts)
        np.testing.assert_array_equal(extra, np.zeros_like(interactions))
        counts[0, 1, 0] = 3
        extra = multiplicity_features(interactions, counts)
        np.testing.assert_array_equal(extra[0, 1, :3], [0.5, 0.5, 0.5])
        self.assertTrue(((extra >= 0) & (extra < 1)).all())
        np.testing.assert_array_equal(
            multiplicity_features(np.zeros_like(interactions), counts),
            np.zeros_like(interactions),
        )
        with self.assertRaisesRegex(ValueError, "integral"):
            multiplicity_features(interactions, counts + 0.5)

    def test_unique_atom_sets_do_not_count_ring_automorphisms(self):
        names = [r.name for r in RULES]
        benzene = motif_counts("c1ccccc1")
        self.assertEqual(benzene[names.index("phenyl_ion")], 1)
        self.assertEqual(benzene[names.index("benzyl_tropylium_ion")], 1)
        self.assertEqual(
            motif_counts("c1ccc(-c2ccccc2)cc1")[names.index("phenyl_ion")], 2
        )

    def test_boolean_presence_matches_existing_structural_support(self):
        for smiles in [
            "CCO",
            "COCOC",
            "CC(=O)O",
            "OP(=O)(O)O",
            "CS(=O)(=O)C",
            "c1ccccc1",
        ]:
            self.assertEqual(
                tuple(np.array(motif_counts(smiles)) > 0), structural_support(smiles)
            )
        with self.assertRaisesRegex(ValueError, "Valid training"):
            motif_counts("not a molecule")


if __name__ == "__main__":
    unittest.main()
