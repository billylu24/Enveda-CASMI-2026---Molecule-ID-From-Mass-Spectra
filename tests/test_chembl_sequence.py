import unittest

from casmi_ml.chembl_sequence_pilot import reorder_scoreable


class ExternalSequenceTests(unittest.TestCase):
    def test_unsupported_candidates_preserve_their_positions(self):
        base = ["unsupported", "a", "long", "b", "c"]
        scores = {
            "a": {"query": 0, "prior": 0},
            "b": {"query": -5, "prior": 0},
            "c": {"query": -1, "prior": -5},
        }
        self.assertEqual(
            reorder_scoreable(base, scores, 1), ["unsupported", "a", "long", "c", "b"]
        )
        self.assertEqual(reorder_scoreable(base, {}, 1), base)
