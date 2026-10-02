import unittest

from casmi_ml.experimental_release import publication_slug


class PublicationSlugTests(unittest.TestCase):
    def test_long_identifiers_are_valid_unique_and_stable(self):
        a = publication_slug("0044_generated_position_update")
        b = publication_slug("0044_generated_position_updates")
        self.assertLessEqual(len("casmi26-research-" + a + "-bundle"), 50)
        self.assertLessEqual(len("CASMI Research " + a + " Assets"), 50)
        self.assertNotEqual(a, b)
        self.assertEqual(a, publication_slug("0044_generated_position_update"))
        self.assertEqual(
            publication_slug("0024_single_decoder_full"), "0024-single-decoder-full"
        )


if __name__ == "__main__":
    unittest.main()
