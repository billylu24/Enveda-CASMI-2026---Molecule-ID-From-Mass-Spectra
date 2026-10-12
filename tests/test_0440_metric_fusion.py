"""Tests for metric identity fusion and window constraint."""

import pytest
from casmi_ml.metric_fusion import metric_fusion


def dummy_identity(smiles: str):
    # Map synonyms to same canonical key
    mapping = {
        "CC(=O)O": "KEY_ACETIC_ACID",
        "CC(=O)[O-]": "KEY_ACETIC_ACID",
        "CCO": "KEY_ETHANOL",
    }
    return mapping.get(smiles, smiles)


def test_metric_fusion_deduplication():
    # first engine has two representations of the same skeleton
    first = ["CC(=O)O", "CC(=O)[O-]"]
    second = ["CCO", "CC(=O)O"]

    res, scores, collapsed = metric_fusion(first, second, dummy_identity, alpha=0.6, limit=40)
    # The duplicate in first should collapse to 1 vote
    assert collapsed >= 1
    # Only unique skeletons should be present
    keys = [dummy_identity(s) for s in res]
    assert len(keys) == len(set(keys))
    # Acetic acid and ethanol are the 2 unique molecules
    assert len(res) == 2


def test_window_limit():
    first = [f"C{i}" for i in range(50)]
    second = [f"N{i}" for i in range(50)]
    res, scores, collapsed = metric_fusion(first, second, lambda s: s, limit=40)
    assert len(res) == 40
