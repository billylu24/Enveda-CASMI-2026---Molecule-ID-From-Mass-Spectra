import unittest

from casmi_ml.generated_fragment_experiment import insert_fragment_ranked


class GeneratedFragmentTests(unittest.TestCase):
    def test_real_evidence_reorders_novel_candidates_and_preserves_prefix(self):
        base = ["A", "B", "C", "D", "E"]
        candidates = [{"key": k} for k in ["B", "F", "G", "G", "H"]]
        ranked = insert_fragment_ranked(
            base, candidates, {"B": 100, "F": 0, "G": 2, "H": 1}, 1
        )
        self.assertEqual(ranked, ["A", "B", "C", "G", "H", "F", "D", "E"])
        self.assertEqual(len(ranked), len(set(ranked)))

    def test_unsupported_tied_and_zero_weight_evidence_preserve_control(self):
        base = ["A", "B", "C", "D"]
        candidates = [{"key": k} for k in ["F", "G", "H"]]
        expected = ["A", "B", "C", "F", "G", "H", "D"]
        for scores, weight in [({}, 1), ({"F": 2, "G": 2, "H": 2}, 1), ({"H": 5}, 0)]:
            self.assertEqual(
                insert_fragment_ranked(base, candidates, scores, weight), expected
            )

    def test_only_five_novel_slots_are_inserted(self):
        base = ["A", "B", "C", "D"]
        candidates = [{"key": k} for k in ["A", "E", "F", "G", "H", "I", "J"]]
        self.assertEqual(
            insert_fragment_ranked(base, candidates, {"J": 10}, 1),
            ["A", "B", "C", "J", "E", "F", "G", "H", "D"],
        )


if __name__ == "__main__":
    unittest.main()
