import unittest

from casmi_ml.chembl_tail_controls import validate_tail


class ProtectedOriginalPrefixTests(unittest.TestCase):
    def test_short_original_keeps_original_but_allows_tail_change(self):
        validate_tail(["a", "b"], ["a", "b", "x"], ["a", "b", "y"], 0.8)

    def test_full_prefix_keeps_first10_but_allows_tail_change(self):
        prior = list(range(15))
        validate_tail(
            prior, prior[:10] + ["x"] + prior[10:], prior[:10] + ["y"] + prior[10:], 0.8
        )

    def test_short_original_disallows_changing_its_existing_candidates(self):
        with self.assertRaisesRegex(ValueError, "protected prefix"):
            validate_tail(["a", "b"], ["a", "b", "x"], ["a", "y", "b"], 0.8)

    def test_low_arm_remains_entirely_frozen(self):
        with self.assertRaisesRegex(ValueError, "low arm"):
            validate_tail(["a"], ["a", "x"], ["a", "y"], 0.49)

    def test_explicit_chemically_supported_prefix3_preserves_original_first3(self):
        validate_tail(
            ["a", "b", "c", "d"],
            ["a", "b", "c", "d", "x"],
            ["a", "b", "c", "y", "d"],
            0.8,
            prefix=3,
        )
        with self.assertRaisesRegex(ValueError, "protected prefix"):
            validate_tail(
                ["a", "b", "c", "d"],
                ["a", "b", "c", "d", "x"],
                ["a", "b", "y", "c", "d"],
                0.8,
                prefix=3,
            )

    def test_high_boundary_uses_original_prefix(self):
        validate_tail(["a"], ["a", "x"], ["a", "y"], 0.5)
