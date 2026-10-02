import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from casmi_ml.graph_candidates import generate_candidates, group_seed


ANCHOR = "CC(C)CCCCC(=O)O"
SECOND = "CCC(C)CCCC(=O)O"
ENUMERATOR = rdMolStandardize.TautomerEnumerator()


def official(smiles):
    return Chem.MolToInchiKey(ENUMERATOR.Canonicalize(Chem.MolFromSmiles(smiles)))[:14]


def degree_signature(mol):
    return sorted((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetDegree(),
                   sum(b.GetBondTypeAsDouble() for b in a.GetBonds())) for a in mol.GetAtoms())


def test_valid_new_graphs_preserve_formula_mass_and_atom_degree():
    rows, report = generate_candidates([ANCHOR, SECOND], {official(ANCHOR), official(SECOND)}, seed=11)
    assert rows and len(rows) <= 32
    assert report["learned_graph_decoder"] is False
    assert len(report["anchor_ids"]) <= 2
    assert report["attempted_edits"] <= 512
    assert len({r["identity"] for r in rows}) == len(rows)
    anchors = {official(s): Chem.MolFromSmiles(s) for s in [ANCHOR, SECOND]}
    for row in rows:
        mol = Chem.MolFromSmiles(row["normalized_smiles"])
        anchor = anchors[row["anchor_identity"]]
        assert mol is not None and len(Chem.GetMolFrags(mol)) == 1
        assert rdMolDescriptors.CalcMolFormula(mol) == rdMolDescriptors.CalcMolFormula(anchor)
        assert abs(Descriptors.ExactMolWt(mol)-Descriptors.ExactMolWt(anchor)) < 1e-8
        assert row["mass"] == pytest.approx(Descriptors.ExactMolWt(anchor), abs=1e-8)
        assert degree_signature(mol) == degree_signature(anchor)
        assert row["identity"] == official(row["normalized_smiles"]) == row["inchikey14"]
        assert row["origin"] == "graph_edit" and row["formal_charge"] == Chem.GetFormalCharge(anchor)
        assert row["identity"] not in anchors


def test_fixed_seed_cache_and_existing_graph_exclusion():
    first, report = generate_candidates([ANCHOR], set(), seed=22)
    repeated, repeated_report = generate_candidates([ANCHOR], set(), seed=22)
    assert first == repeated and repeated_report["cache_hits"] >= 1
    excluded = {row["identity"] for row in first}
    filtered, stats = generate_candidates([ANCHOR], excluded, seed=22)
    assert not (excluded & {row["identity"] for row in filtered})
    assert stats["excluded_existing"] >= len(excluded)


def test_invalid_small_duplicate_and_zero_budget_anchors():
    assert generate_candidates(["bad smiles", "C", "CC"], set())[0] == []
    rows, report = generate_candidates([ANCHOR, ANCHOR, SECOND, "CCCCCCCCCC"], set(), max_new=3)
    assert len(rows) <= 3 and report["anchors_used"] == 2
    assert generate_candidates([ANCHOR], set(), max_new=0)[0] == []
    assert generate_candidates([ANCHOR], set(), attempts=0)[0] == []
    with pytest.raises(ValueError):
        generate_candidates([ANCHOR], set(), max_new=33)


def test_query_seed_uses_content_not_truth_or_group_order():
    rows = [{"precursor_mz": 175., "adduct": "[M+H]+", "ionization_mode": "positive",
             "ms2_mzs": [55., 70.], "ms2_normalized_intensities": [1., .5], "identity": "truth-A"},
            {"precursor_mz": 175., "adduct": "[M+H]+", "ionization_mode": "positive",
             "ms2_mzs": [60.], "ms2_normalized_intensities": [1.], "identity": "truth-B"}]
    frame = pd.DataFrame(rows)
    seed = group_seed(frame)
    assert seed == group_seed(frame.iloc[::-1])
    frame["identity"] = "different-truth"
    frame["molecule_id"] = "different-query-label"
    assert seed == group_seed(frame)
    frame.loc[0, "precursor_mz"] = 176.
    assert seed != group_seed(frame)
