"""Guards against silently reusing scores or invalid train-only negatives."""

import numpy as np
import pytest

from casmi_ml.chembl_critic_slots import score_cache_key
from casmi_ml.encoder_pair_finetuning import validate_hard_table


def test_proposal_score_cache_binds_both_models():
    args = ("query", "first", ["candidate"])
    original = {"encoder_sha256": "encoder1", "critic_sha256": "critic1"}
    keys = {
        score_cache_key(*args),
        score_cache_key(*args, original),
        score_cache_key(*args, dict(original, encoder_sha256="encoder2")),
        score_cache_key(*args, dict(original, critic_sha256="critic2")),
    }
    assert len(keys) == 4
    assert score_cache_key(*args, original) == score_cache_key(*args, original.copy())


def negative_fixture():
    count = 17
    candidates = np.asarray([[j for j in range(count) if j != i] for i in range(count)])
    return candidates[:, :15].copy(), {
        "keys": list(range(count)),
        "negatives": candidates,
        "inside": np.full(count, 16),
    }


def test_hard_negative_duplicates_positive_and_foreign_pool_rejected():
    hard, data = negative_fixture()
    validate_hard_table(hard, data)
    duplicate = hard.copy()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(ValueError, match="Invalid training-only"):
        validate_hard_table(duplicate, data)
    positive = hard.copy()
    positive[0, 0] = 0
    with pytest.raises(ValueError, match="Invalid training-only"):
        validate_hard_table(positive, data)
    wrong_pool = dict(data, negatives=data["negatives"].copy())
    wrong_pool["negatives"][0, 0] = 0
    with pytest.raises(ValueError, match="outside frozen"):
        validate_hard_table(hard, wrong_pool)
