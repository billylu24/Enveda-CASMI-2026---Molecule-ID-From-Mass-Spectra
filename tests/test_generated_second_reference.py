import unittest

from casmi_ml.generated_second_reference import select_prefix


class SecondReferenceTests(unittest.TestCase):
    def test_second_reference_is_protected_independent_of_first(self):
        self.assertEqual(
            select_prefix(["A", "B"], {"B"}, 0.1, "second_unreferenced"), 2
        )
        self.assertEqual(
            select_prefix(["A", "B"], {"A"}, 0.1, "second_unreferenced"), 1
        )
        self.assertEqual(
            select_prefix(["A", "B"], {"A"}, 0.9, "second_unreferenced_lowconf"), 2
        )
        self.assertEqual(
            select_prefix(["A", "B"], {"A"}, 0.1, "second_unreferenced_lowconf"), 1
        )
        self.assertEqual(select_prefix(["A"], set(), 0.1, "second_unreferenced"), 2)
        self.assertEqual(select_prefix(["A", "B"], set(), 0.1, "baseline"), 2)


if __name__ == "__main__":
    unittest.main()
