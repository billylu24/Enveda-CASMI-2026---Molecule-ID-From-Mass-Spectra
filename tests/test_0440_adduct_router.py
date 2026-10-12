"""Tests for adduct routing and policy module."""

import pytest
from casmi_ml.adduct_router import (
    AdductClass,
    IonMode,
    classify_adduct,
    summarize_molecule_adducts,
)


def test_classify_standard_protonated():
    policy = classify_adduct("[M+H]+", mode_val=1)
    assert policy.adduct == "[M+H]+"
    assert policy.ion_mode == IonMode.POSITIVE
    assert policy.adduct_class == AdductClass.PROTONATED
    assert policy.glacier_eligible is True
    assert policy.popularity_prior_weight == 0.25


def test_classify_standard_deprotonated():
    policy = classify_adduct("[M-H]-", mode_val=-1)
    assert policy.adduct == "[M-H]-"
    assert policy.ion_mode == IonMode.NEGATIVE
    assert policy.adduct_class == AdductClass.DEPROTONATED
    assert policy.glacier_eligible is False
    assert policy.iceberg_eligible is True


def test_classify_sodiated():
    policy = classify_adduct("[M+Na]+", mode_val=1)
    assert policy.adduct == "[M+Na]+"
    assert policy.ion_mode == IonMode.POSITIVE
    assert policy.adduct_class == AdductClass.SODIATED
    assert policy.glacier_eligible is False
    assert policy.popularity_prior_weight == 0.35  # Higher prior for sodiated


def test_summarize_molecule_adducts():
    spectra = [
        {"adduct": "[M+Na]+", "mode": 1},
        {"adduct": "[M+H]+", "mode": 1},
    ]
    summary = summarize_molecule_adducts(spectra)
    assert summary["adduct_count"] == 2
    assert summary["has_m_plus_h"] is True
    assert summary["glacier_eligible"] is True

    neg_spectra = [
        {"adduct": "[M-H]-", "mode": -1},
    ]
    neg_summary = summarize_molecule_adducts(neg_spectra)
    assert neg_summary["has_m_minus_h"] is True
    assert neg_summary["glacier_eligible"] is False
    assert neg_summary["primary_mode"] == "negative"
