import unittest

from casmi_ml.chembl_fragment_pilot import (
    informative_fragments,
    possible_critic_gate,
    supported_proposals,
)


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


class SupportedProposalTests(unittest.TestCase):
    def test_each_insertion_requires_its_own_critic_and_fragment_support(self):
        proposed = ["a", "b", "c", "d", "e"]
        critic = {"first": 0.5, "a": 0.6, "b": 0.55, "c": 0.7, "d": 0.65, "e": 0.8}
        fragments = {"a": 2, "b": 3, "c": 0, "d": 1}
        self.assertEqual(
            supported_proposals(proposed, critic, "first", fragments, 0.05), ["a", "d"]
        )


class PossibleCriticGateTests(unittest.TestCase):
    def test_no_fragment_order_can_pass_when_max_critic_cannot_pass(self):
        scores = {"first": 0.5, "a": 0.55, "b": 0.2, "c": 0.6}
        self.assertFalse(possible_critic_gate(["a", "b"], scores, "first", 0.05))
        self.assertTrue(possible_critic_gate(["a", "c"], scores, "first", 0.05))
