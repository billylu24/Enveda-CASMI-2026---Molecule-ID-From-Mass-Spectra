import unittest

from casmi_ml.generated_score_combination import combined_order


class GeneratedScoreCombinationTests(unittest.TestCase):
    def test_combination_uses_observed_fields_and_keeps_control(self):
        candidates = [
            {
                "key": "A",
                "smiles": "CC",
                "log_probability": -2,
                "chemical_score": 0,
                "formula_support": 0,
                "best_sequence_tokens": 3,
                "sample_count": 1,
            },
            {
                "key": "B",
                "smiles": "CCC",
                "log_probability": -4,
                "chemical_score": 0,
                "formula_support": 0,
                "best_sequence_tokens": 9,
                "sample_count": 5,
            },
        ]
        self.assertEqual(combined_order(candidates, {"B": 1}, (0, 0, 0)), ["A", "B"])
        self.assertEqual(combined_order(candidates, {"B": 1}, (1, 1, 0.5)), ["B", "A"])
        self.assertEqual(combined_order([], {}, (1, 1, 0.5)), [])


if __name__ == "__main__":
    unittest.main()
