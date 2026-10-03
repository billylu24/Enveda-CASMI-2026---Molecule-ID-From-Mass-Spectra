import unittest

from casmi_ml.low_fragment_first import promote_low_leader


class LowLeaderTests(unittest.TestCase):
    def test_only_leader_moves_and_all_other_relative_order_is_preserved(self):
        original = ["a", "b", "c", "d"]
        selected = ["a", "b", "c", "x", "y", "z", "d"]
        result = promote_low_leader(original, selected)
        self.assertEqual(result, ["x", "a", "b", "c", "y", "z", "d"])
        self.assertEqual(
            [k for k in result if k != "x"], [k for k in selected if k != "x"]
        )

    def test_short_original_list_has_no_invented_protected_prefix(self):
        self.assertEqual(promote_low_leader(["a"], ["a", "x", "y"]), ["x", "a", "y"])

    def test_any_unrelated_original_reordering_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            promote_low_leader(["a", "b", "c"], ["b", "a", "c", "x"])


class ObservableLowLeaderTests(unittest.TestCase):
    def test_strict_fragment_and_critic_gates_and_fallbacks(self):
        from casmi_ml.low_leader_inference import select_low_fragment

        current = ["a", "b", "c", "d"]
        for fragments, critic, fallback in [
            ({"x": 2, "y": 0, "a": 2}, {"x": 0.6, "a": 0.4}, False),
            ({"x": 2, "y": 0, "a": 1}, {"x": 0.45, "a": 0.4}, False),
            ({"x": 2, "y": 0, "a": 1}, {"x": 0.6, "a": 0.4}, True),
            ({"x": 2, "y": 2, "a": 1}, {"x": 0.6, "a": 0.4}, False),
        ]:
            result, inserted, _ = select_low_fragment(
                current, ["x", "y"], critic, fragments, fallback, True
            )
            self.assertEqual(result, current)
            self.assertEqual(inserted, [])

    def test_legacy_default_and_optional_leader_share_eligibility(self):
        from casmi_ml.low_leader_inference import select_low_fragment

        args = (
            ["a", "b", "c", "d"],
            ["x", "y"],
            {"x": 0.6, "a": 0.4},
            {"x": 2, "y": 0, "a": 1},
            False,
        )
        old, _, status = select_low_fragment(*args)
        new, _, new_status = select_low_fragment(*args, leader_first=True)
        self.assertEqual(status, "low_fragment_inserted")
        self.assertEqual(new_status, "low_supported_leader_first")
        self.assertEqual(new, promote_low_leader(args[0], old))
