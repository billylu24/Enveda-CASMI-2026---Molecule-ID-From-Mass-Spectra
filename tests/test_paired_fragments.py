import unittest
from unittest.mock import patch

from casmi_ml.paired_fragments import score_group_mean


class Fragmenter:
    timeout = 60

    def score(self, row, candidates):
        return row


class PairedFragmentTests(unittest.TestCase):
    def test_instrument_scale_does_not_dominate_relative_comparison(self):
        fragmenter = Fragmenter()
        rows = [
            {"status": "complete", "scores": {"first": 100, "new": 10}},
            {"status": "complete", "scores": {"first": 1, "new": 10}},
            {"status": "unsupported_ion_or_resolution", "scores": {}},
            {"status": "complete", "scores": {"first": 50, "new": 50}},
        ]
        scores, fallback = score_group_mean(fragmenter, rows, ["first", "new"])
        self.assertFalse(fallback)
        self.assertAlmostEqual(scores["first"], 0.55)
        self.assertAlmostEqual(scores["new"], 0.55)
        self.assertEqual(fragmenter.timeout, 60)

    def test_invalid_or_noninformative_evidence_never_enables_insertion(self):
        for scores in ({}, {"a": 0, "b": 0}, {"a": float("nan")}, {"a": -1}):
            self.assertEqual(
                score_group_mean(
                    Fragmenter(), [{"status": "complete", "scores": scores}], ["a", "b"]
                ),
                ({}, False),
            )

    def test_budget_exhaustion_discards_partial_group_and_restores_timeout(self):
        fragmenter = Fragmenter()
        with patch("casmi_ml.paired_fragments.time.monotonic", side_effect=[0, 2]):
            scores, fallback = score_group_mean(
                fragmenter,
                [{"status": "complete", "scores": {"a": 2, "b": 1}}],
                ["a", "b"],
                deadline=1,
            )
        self.assertEqual(scores, {})
        self.assertTrue(fallback)
        self.assertEqual(fragmenter.timeout, 60)
