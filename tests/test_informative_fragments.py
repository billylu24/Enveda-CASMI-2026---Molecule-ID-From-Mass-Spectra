import unittest

from casmi_ml.chembl_fragment_pilot import informative_fragments


class InformativeFragmentTests(unittest.TestCase):
    def test_missing_tied_and_unscored_winner_preserve_current_ranking(self):
        for scores in (
            {},
            {"a": 0, "b": 0},
            {"a": 3, "b": 3},
            {"b": 3},
            {"a": float("nan"), "b": 1},
        ):
            self.assertFalse(informative_fragments(["a", "b"], scores))

    def test_distinct_positive_scored_winner_allows_evidence_branch(self):
        self.assertTrue(informative_fragments(["a", "b"], {"a": 3, "b": 1}))
        self.assertTrue(informative_fragments(["a", "b"], {"a": 3}))
