import unittest

from casmi_ml.catalog_scoring_funnel_audit import bucket, missing_query_keys, rr


class CatalogScoringFunnelTest(unittest.TestCase):
    def test_exclusive_buckets_preserve_current_priority(self):
        self.assertEqual(bucket("a", ["a"], {"a"}, {"a"}, {"a"}), "original_first")
        self.assertEqual(
            bucket("a", ["b", "a"], {"a"}, {"a"}, {"a"}), "original_current_other"
        )
        self.assertEqual(bucket("a", ["b"], {"a"}, {"a"}, {"a"}), "external_scored")
        self.assertEqual(
            bucket("a", ["b"], set(), {"a"}, {"a"}), "valid_mass_window_unscored"
        )
        self.assertEqual(
            bucket("a", [], set(), set(), {"a"}), "catalog_outside_valid_mass_window"
        )
        self.assertEqual(
            bucket("a", [], set(), set(), set()),
            "absent_external_catalog_and_original_current",
        )

    def test_full_ranking_tail_is_not_top25(self):
        ranking = list(range(30))
        self.assertEqual(rr(ranking, 24), 1 / 25)
        self.assertEqual(rr(ranking, 25), 0)
        self.assertEqual(rr([], "absent"), 0)

    def test_missing_observable_rows_stay_in_complete_denominator(self):
        self.assertEqual(
            missing_query_keys({"a": []}, {"a": ["a"], "b": []}, {"a": ["a"], "b": []}),
            {"b"},
        )
        with self.assertRaises(ValueError):
            missing_query_keys(
                {"a": []}, {"a": ["a"], "b": ["b"]}, {"a": ["a"], "b": []}
            )
        with self.assertRaises(ValueError):
            missing_query_keys({"a": [], "extra": []}, {"a": ["a"]}, {"a": ["a"]})
