import unittest

from casmi_ml.generated_expanded_first import expanded_prefix
from casmi_ml.generation_slots import insert_generated


class ExpandedFirstTests(unittest.TestCase):
    def test_supported_first_protected_and_unsupported_rule_explicit(self):
        self.assertEqual(expanded_prefix(["a", "b"], {"a"}, "unsupported_first0"), 2)
        self.assertEqual(expanded_prefix(["a", "b"], set(), "unsupported_first0"), 0)
        self.assertEqual(expanded_prefix([], set(), "unsupported_first0"), 2)
        self.assertEqual(expanded_prefix(["a"], set(), "baseline"), 2)
        self.assertEqual(expanded_prefix(["a"], set(), "unsupported_first1"), 1)
        with self.assertRaises(ValueError):
            expanded_prefix(["a"], set(), "invalid")

    def test_zero_prefix_uses_novel_deduplicated_candidates_and_retains_base_order(
        self,
    ):
        self.assertEqual(
            insert_generated(["a", "b"], ["a", "new", "new", "other"], 0, 1),
            ["new", "a", "b"],
        )
        self.assertEqual(insert_generated(["a", "b"], [], 0, 5), ["a", "b"])
        with self.assertRaises(ValueError):
            insert_generated(["a"], ["new"], -1, 1)
