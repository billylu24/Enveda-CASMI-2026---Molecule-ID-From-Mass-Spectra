import unittest

from casmi_ml.low_leader_prefix_evaluation import supported_over_prefix


class LowLeaderPrefixTests(unittest.TestCase):
    def test_leader_must_beat_every_original_prefix_member(self):
        self.assertTrue(
            supported_over_prefix(
                "x", ["a", "b", "c", "d"], {"x": 4, "a": 1, "b": 2, "c": 3}, False
            )
        )
        self.assertFalse(
            supported_over_prefix(
                "x", ["a", "b", "c"], {"x": 3, "a": 1, "b": 2, "c": 3}, False
            )
        )
        self.assertFalse(
            supported_over_prefix("x", ["a", "b", "c"], {"x": 4, "a": 1, "b": 2}, False)
        )
        self.assertFalse(supported_over_prefix("x", ["a"], {"x": 4, "a": 1}, True))

    def test_short_original_requires_only_actual_prefix_and_no_tail(self):
        self.assertTrue(supported_over_prefix("x", ["a"], {"x": 4, "a": 1}, False))
        self.assertTrue(
            supported_over_prefix(
                "x",
                ["a", "b", "c", "d"],
                {"x": 4, "a": 1, "b": 2, "c": 3, "d": 9},
                False,
            )
        )
