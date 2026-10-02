import unittest

from casmi_ml.generated_critic import critic_order


class GeneratedCriticTests(unittest.TestCase):
    def test_frozen_score_order_and_unsupported_fallback(self):
        candidates = [{"key": "A"}, {"key": "B"}, {"key": "C"}]
        self.assertEqual(
            critic_order(candidates, {"A": 0.1, "B": 0.9, "C": 0.9}, 1), ["B", "C", "A"]
        )
        self.assertEqual(critic_order(candidates, {}, 1), ["A", "B", "C"])
        self.assertEqual(critic_order(candidates, {"C": 1}, 0), ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
