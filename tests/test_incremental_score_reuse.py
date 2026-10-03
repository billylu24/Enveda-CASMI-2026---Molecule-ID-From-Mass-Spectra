import unittest

from casmi_ml.incremental_metfrag_benchmark import reuse_scores


class IncrementalScoreReuseTests(unittest.TestCase):
    def test_preserves_raw_scores_and_does_not_invent_absent_zero_scores(self):
        result = reuse_scores(
            {"old": 80.43, "removed": 9.7}, {"old", "new", "omitted"}, {"new": 220.58}
        )
        self.assertEqual(result, {"old": 80.43, "new": 220.58})
        self.assertNotIn("omitted", result)

    def test_rejects_overwrite_of_completed_score(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            reuse_scores({"old": 80.43}, {"old"}, {"old": 80.43})

    def test_rejects_unrequested_incremental_scores(self):
        with self.assertRaisesRegex(ValueError, "unrequested"):
            reuse_scores({}, {"new"}, {"different": 220.58})
