import unittest

from casmi_ml.training_alias_radius_audit import radius_signatures
from casmi_ml.training_fingerprint_alias_audit import representation


class TrainingAliasRadiusTest(unittest.TestCase):
    def test_larger_bits_keep_a_local_alias_but_radius3_separates_it(self):
        left, right = "CCCCCC=CCCCCCCC", "CCCCCCC=CCCCCCC"
        a, b = representation(left), representation(right)
        self.assertNotEqual(a["canonical_graph"], b["canonical_graph"])
        self.assertEqual(a["bits4096"], b["bits4096"])
        self.assertEqual(a["bits8192"], b["bits8192"])
        x, y = radius_signatures(left), radius_signatures(right)
        self.assertEqual(x[2], y[2])
        self.assertNotEqual(x[3], y[3])

    def test_signatures_ignore_atom_order_and_stereochemistry(self):
        self.assertEqual(radius_signatures("CCO"), radius_signatures("OCC"))
        self.assertEqual(
            radius_signatures("C[C@H](O)CC"),
            radius_signatures("C[C@@H](O)CC"),
        )
        with self.assertRaises(ValueError):
            radius_signatures("invalid")
