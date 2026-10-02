import unittest

from casmi_ml.generated_first_critic import promotion_prefix


class CriticPromotionTests(unittest.TestCase):
    def test_strict_observable_margin_ignores_existing_candidates_and_missing_scores(
        self,
    ):
        ranking = ["a", "b"]
        generated = ["b", "new"]
        scores = {"a": 0.4, "b": 0.99, "new": 0.6}
        self.assertEqual(promotion_prefix(ranking, generated, scores, 2, 0.05), 0)
        self.assertEqual(promotion_prefix(ranking, generated, scores, 2, 0.2), 2)
        self.assertEqual(promotion_prefix(ranking, generated, scores, 2, None), 2)
        self.assertEqual(promotion_prefix(ranking, ["a", "b"], scores, 2, 0), 2)
        self.assertEqual(promotion_prefix(ranking, generated, {"new": 0.9}, 2, 0), 2)
