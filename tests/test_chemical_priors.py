import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi_ml.chemical_priors import candidate_scores, extract_evidence, load_rules, rerank


DICTIONARY = Path(__file__).resolve().parents[1] / 'configs/chemical_priors.json'


def spectrum(mz, precursor=500., mode='positive', adduct='[M+H]+', intensities=None):
    return {'ms2_mzs': mz, 'ms2_normalized_intensities': intensities or [1.] * len(mz),
            'precursor_mz': precursor, 'ionization_mode': mode, 'adduct': adduct}


def test_exact_ion_gates_mode_and_adduct_and_matches_choline_motif():
    rules = load_rules(DICTIONARY)
    group = pd.DataFrame([spectrum([184.0733209])])
    evidence = extract_evidence(group, rules)
    assert [e['rule_id'] for e in evidence] == ['phosphocholine_184_positive']
    smiles = {'pc': 'CCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCC',
              'pe': 'CCCC(=O)OCC(COP(=O)(O)OCCN)OC(=O)CCC', 'unrelated': 'c1ccccc1'}
    scores, supported = candidate_scores(smiles, evidence, rules)
    assert scores['pc'] == 1 and scores['pe'] == 0 and scores['unrelated'] == 0
    assert supported['pc'] == ['phosphocholine_184_positive']
    assert extract_evidence([spectrum([184.0733209], mode='negative', adduct='[M-H]-')], rules) == []
    assert extract_evidence([spectrum([184.0733209], adduct='[M+Na]+')], rules) == []


def test_losses_use_ion_precursor_not_neutral_mass():
    rules = load_rules(DICTIONARY)
    loss = 162.0528234
    observed = extract_evidence([spectrum([500. - loss])], rules)
    assert [e['rule_id'] for e in observed] == ['o_hexoside_loss_162']
    assert observed[0]['observations'][0]['mass_error_da'] == pytest.approx(0)
    wrong = extract_evidence([spectrum([500. - loss - 1.0072764666])], rules)
    assert wrong == []
    smiles = {'glycoside': 'COC1OC(CO)C(O)C(O)C1O',
              'free_sugar': 'OC1OC(CO)C(O)C(O)C1O', 'alkane': 'CCCCCC'}
    scores, _ = candidate_scores(smiles, observed, rules)
    assert scores['glycoside'] == 1
    assert scores['free_sugar'] == scores['alkane'] == 0


def test_glucuronide_and_sulfate_motifs():
    rules = load_rules(DICTIONARY)
    for loss, compatible, wrong in [
        (176.0320880, 'COC1OC(C(=O)O)C(O)C(O)C1O', 'COC1OC(CO)C(O)C(O)C1O'),
        (79.9568150, 'COS(=O)(=O)O', 'CS(=O)(=O)O'),
        (141.0190946, 'COP(=O)(O)OCCN', 'COP(=O)(O)OCC[N+](C)(C)C'),
    ]:
        evidence = extract_evidence([spectrum([500.-loss])], rules)
        scores, _ = candidate_scores({'yes': compatible, 'no': wrong}, evidence, rules)
        assert scores == {'yes': 1., 'no': 0.}


def test_repeated_noisy_peaks_cannot_multiply_support():
    rules = load_rules(DICTIONARY)
    one = extract_evidence([spectrum([184.0733209])], rules)
    many = extract_evidence([spectrum([184.0733209, 184.0733209, 184.074])]*6, rules)
    assert len(many) == 1 and many[0]['matched_spectra'] == 6
    assert many[0]['strength'] == one[0]['strength']
    assert extract_evidence([spectrum([184.1])], rules) == []
    assert extract_evidence([spectrum([184.0733209, 200], intensities=[.001, 1])], rules) == []


def test_no_evidence_no_filter_and_rank_ties_preserve_order():
    rules = load_rules(DICTIONARY)
    scores, _ = candidate_scores({'b': 'CCO', 'a': 'COC'}, [], rules)
    assert rerank(['b', 'a'], scores, .4) == ['b', 'a']
    assert rerank(['b', 'a'], {'a': 1, 'b': 0}, 0) == ['b', 'a']
    output = rerank(['b', 'a', 'c'], {'a': 1}, .8)
    assert output[0] == 'a' and set(output) == {'b', 'a', 'c'}
    assert output.index('b') < output.index('c')


def test_weak_peak_reduces_fusion_and_n_methyl_pe_is_not_141_loss():
    rules = load_rules(DICTIONARY)
    evidence = extract_evidence([spectrum([500.-141.0190946])], rules)
    scores, _ = candidate_scores({'primary': 'COP(=O)(O)OCCN',
                                  'protonated': 'COP(=O)([O-])OCC[NH3+]',
                                  'methylated': 'COP(=O)(O)OCCNC'}, evidence, rules)
    assert scores['primary'] == scores['protonated'] == 1
    assert scores['methylated'] == 0
    weak = extract_evidence([spectrum([184.0733209, 200.], intensities=[.01, 1.])], rules)
    strong = extract_evidence([spectrum([184.0733209])], rules)
    structures = {'pc': 'COP(=O)([O-])OCC[N+](C)(C)C', 'other': 'CCCC'}
    weak_scores, _ = candidate_scores(structures, weak, rules)
    strong_scores, _ = candidate_scores(structures, strong, rules)
    assert weak_scores['pc'] == pytest.approx(.1)
    assert strong_scores['pc'] == 1
    assert rerank(['other', 'pc'], weak_scores, .8) == ['other', 'pc']
    assert rerank(['other', 'pc'], strong_scores, .8) == ['pc', 'other']


def test_unsupported_adduct_empty_nan_and_alignment():
    rules = load_rules(DICTIONARY)
    assert extract_evidence([spectrum([184.0733209], adduct='[M+2H]2+')], rules) == []
    assert extract_evidence([spectrum([])], rules) == []
    assert extract_evidence([spectrum([np.nan])], rules) == []
    with pytest.raises(ValueError):
        extract_evidence([spectrum([184.0733209], intensities=[1, 2])], rules)
    for options in [{'ppm': float('inf')}, {'absolute_tolerance': float('nan')}]:
        with pytest.raises(ValueError):
            extract_evidence([spectrum([184.0733209])], rules, **options)


def test_invalid_dictionary_and_weights_fail_fast(tmp_path):
    raw = json.loads(DICTIONARY.read_text())
    raw['rules'][0]['smarts'] = ['invalid(']
    path = tmp_path / 'rules.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_rules(path)
    for weight in [-1, 1.1, float('nan')]:
        with pytest.raises(ValueError):
            rerank(['a'], {}, weight)
