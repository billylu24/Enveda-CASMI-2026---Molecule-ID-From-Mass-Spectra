"""Guards against silently reusing scores or invalid train-only negatives."""

import unittest

import numpy as np

from casmi_ml.chembl_critic_slots import score_cache_key, validate_proposal_membership
from casmi_ml.chembl_fragment_pilot import score_cache_key as fragment_cache_key
from casmi_ml.encoder_pair_finetuning import validate_hard_table
from casmi_ml.chembl_prior_proposals import ordered_proposals


class EncoderPairBindingTests(unittest.TestCase):
    def test_proposal_score_cache_binds_both_models(self):
        args = ("query", "first", ["candidate"])
        original = {"encoder_sha256": "encoder1", "critic_sha256": "critic1"}
        keys = {
            score_cache_key(*args),
            score_cache_key(*args, original),
            score_cache_key(*args, dict(original, encoder_sha256="encoder2")),
            score_cache_key(*args, dict(original, critic_sha256="critic2")),
        }
        self.assertEqual(len(keys), 4)
        self.assertEqual(
            fragment_cache_key(*args, original), score_cache_key(*args, original)
        )
        self.assertNotEqual(
            fragment_cache_key(*args), fragment_cache_key(*args, original)
        )
        self.assertEqual(
            score_cache_key(*args, original), score_cache_key(*args, original.copy())
        )

    def test_preselection_ties_use_the_scored_candidate_keys(self):
        self.assertEqual(
            ordered_proposals(["z", "a", "b"], np.zeros(3)), ["a", "b", "z"]
        )
        self.assertEqual(
            ordered_proposals(["z", "a", "b"], np.array([0.0, 1.0, 0.0])),
            ["a", "b", "z"],
        )
        with self.assertRaises(ValueError):
            ordered_proposals(["a"], np.array([np.nan]))

    def test_preselection_cannot_change_mass_pool_membership(self):
        original = {"q": ["a", "b"]}
        validate_proposal_membership({"q": ["b", "a"]}, original)
        for changed in ({"q": ["a", "c"]}, {"q": ["a"]}, {}, {"q": ["a", "a"]}):
            with self.assertRaisesRegex(ValueError, "original mass window"):
                validate_proposal_membership(changed, original)

    def test_invalid_negatives_rejected(self):
        count = 17
        candidates = np.asarray(
            [[j for j in range(count) if j != i] for i in range(count)]
        )
        hard = candidates[:, :15].copy()
        data = {
            "keys": list(range(count)),
            "negatives": candidates,
            "inside": np.full(count, 16),
        }
        validate_hard_table(hard, data)
        duplicate = hard.copy()
        duplicate[0, 1] = duplicate[0, 0]
        with self.assertRaisesRegex(ValueError, "Invalid training-only"):
            validate_hard_table(duplicate, data)
        positive = hard.copy()
        positive[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "Invalid training-only"):
            validate_hard_table(positive, data)
        wrong_pool = dict(data, negatives=data["negatives"].copy())
        wrong_pool["negatives"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "outside frozen"):
            validate_hard_table(hard, wrong_pool)
