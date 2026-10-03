import unittest

from casmi_ml.chembl_high_prefix1 import promoted_slots


class HighPrefix1Tests(unittest.TestCase):
    def test_first_and_existing_relative_order_protected_same_three_slots(self):
        prior = ["first", "b", "c", "d"]
        order = promoted_slots(prior, ["x", "y", "z", "other"])
        self.assertEqual(order, ["first", "x", "y", "z", "b", "c", "d"])
        self.assertEqual([k for k in order if k in prior], prior)

    def test_short_original_and_no_novel_candidates(self):
        self.assertEqual(promoted_slots(["first"], ["x"]), ["first", "x"])
        self.assertEqual(promoted_slots(["first", "b"], ["b"]), ["first", "b"])
