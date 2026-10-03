import unittest

from casmi_ml.chembl_high_slot_support import supported_slots


class HighSlotSupportTests(unittest.TestCase):
    def test_each_of_three_slots_requires_both_strict_margins(self):
        prior = ["first", "b", "c", "d"]
        critic = {"first": 0.0, "x": 0.1, "y": 0.05, "z": 0.2, "other": 1.0}
        fragments = {"first": 1.0, "x": 2.0, "y": 2.0, "z": 1.0, "other": 10.0}
        self.assertEqual(
            supported_slots(prior, ["x", "y", "z", "other"], critic, fragments),
            ["first", "b", "c", "x", "d"],
        )

    def test_actual_short_prefix_and_no_supported_slots(self):
        prior = ["first"]
        self.assertEqual(
            supported_slots(
                prior, ["x"], {"first": 0.0, "x": 0.2}, {"first": 1.0, "x": 2.0}
            ),
            ["first", "x"],
        )
        self.assertEqual(
            supported_slots(prior, ["x"], {"first": 0.0, "x": 0.2}, {}), prior
        )
