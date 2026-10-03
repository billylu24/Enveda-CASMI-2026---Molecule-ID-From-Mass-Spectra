import unittest

from casmi_ml.chembl_high_fragment_pilot import fragment_structures


class ActualFirstRepresentationTests(unittest.TestCase):
    def test_generated_actual_first_takes_precedence_over_catalog_and_reference(self):
        result = fragment_structures(
            "query",
            "first",
            ["new"],
            {"first": "catalog", "new": "candidate"},
            {"first": "reference"},
            {"query": {"first": "generated"}},
        )
        self.assertEqual(result, {"first": "generated", "new": "candidate"})

    def test_original_reference_takes_precedence_over_catalog(self):
        result = fragment_structures(
            "query",
            "first",
            ["new"],
            {"first": "catalog", "new": "candidate"},
            {"first": "reference"},
            {"query": {}},
        )
        self.assertEqual(result["first"], "reference")

    def test_missing_actual_first_rejects_catalog_substitution(self):
        with self.assertRaisesRegex(ValueError, "representation missing"):
            fragment_structures(
                "query", "first", [], {"first": "catalog"}, {}, {"query": {}}
            )
