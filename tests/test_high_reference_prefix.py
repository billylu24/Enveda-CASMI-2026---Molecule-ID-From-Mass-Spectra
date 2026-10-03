import unittest

from casmi_ml.chembl_high_reference_prefix import (
    promoted_slots,
    reference_supported_prefix,
)


class HighReferencePrefixTests(unittest.TestCase):
    def test_later_observed_original_positions_protected_with_same_new_slots(self):
        prior = ["first", "b", "c", "d"]
        self.assertEqual(reference_supported_prefix(prior, {"b"}), 2)
        self.assertEqual(reference_supported_prefix(prior, {"c"}), 3)
        self.assertEqual(reference_supported_prefix(prior, {"first", "d"}), 1)
        self.assertEqual(
            promoted_slots(prior, ["x", "y", "z", "other"], {"b"}),
            ["first", "b", "x", "y", "z", "c", "d"],
        )

    def test_short_original_and_no_query_identity_or_confidence_argument(self):
        self.assertEqual(promoted_slots(["first"], ["x"], set()), ["first", "x"])
        self.assertEqual(
            promoted_slots(["first", "b"], ["x"], {"b"}), ["first", "b", "x"]
        )
