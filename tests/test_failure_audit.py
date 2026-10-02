import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from baseline import PROTON
from casmi_ml.chemical_priors import load_rules
from casmi_ml.failure_audit import (
    audit_query, canonical_identity, prepare_catalog, rule_gate_audit,
    save_audit, summarize_behavior, summarize_cases,
)


def catalog(smiles, origin="chebi", mass=None):
    return pd.DataFrame([{"normalized_smiles": smiles, "mass": mass if mass is not None else Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)), "origin": origin}])


def query(smiles="CCO", offset=0., mode="positive", peaks=None, intensity=None):
    return pd.DataFrame([{"molecule_id": "query", "normalized_smiles": smiles,
                          "precursor_mz": Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)) + PROTON + offset,
                          "ionization_mode": mode, "adduct": "[M+H]+",
                          "ms2_mzs": peaks if peaks is not None else [20.],
                          "ms2_normalized_intensities": intensity if intensity is not None else [1.]}])


class FailureAuditTests(unittest.TestCase):
    def test_absent_pool_mass_miss_and_ranking_miss_are_distinct(self):
        absent = audit_query(query(), catalog("CCN"), [], ())
        self.assertEqual(absent["failure_mechanism"], "truth_absent_from_supplied_candidate_pool")
        mass = audit_query(query(offset=.1), catalog("CCO"), [], ())
        self.assertTrue(mass["truth_in_supplied_pool"])
        self.assertEqual(mass["failure_mechanism"], "mass_filter_miss")
        rank = audit_query(query(), catalog("CCO"), [("other", "COC")], ())
        self.assertTrue(rank["truth_mass_eligible"])
        self.assertEqual(rank["failure_mechanism"], "ranking_or_top25_truncation")

    def test_low_mass_absolute_floor_and_canonical_alias_are_preserved(self):
        case = audit_query(query("CC=O", offset=.0059), catalog("C=CO"), [("alias", "C=CO")], ())
        self.assertEqual(canonical_identity("CC=O"), canonical_identity("C=CO"))
        self.assertTrue(case["truth_mass_eligible"])
        self.assertEqual(case["true_rank"], 1)
        self.assertEqual(case["mass_window_da"], .006)

    def test_wrong_guard_is_observed_without_claiming_a_cure(self):
        case = audit_query(query(), catalog("CCO"), [("other", "COC"), ("truth", "CCO")], (), confidence=.95)
        self.assertTrue(case["guard_wrong_top1"])
        self.assertFalse(case["guard_protected_top25_miss"])
        self.assertEqual(case["true_rank"], 2)
        self.assertEqual(case["failure_mechanism"], "ranking_not_top1")

    def test_all_query_rules_run_even_when_protected_and_encoding_alias_is_diagnostic(self):
        rules = load_rules(Path(__file__).resolve().parents[1] / "configs/chemical_priors.json")
        pc = "C[N+](C)(C)CCOP(=O)([O-])OCC"
        group = query(pc, peaks=[184.0733209])
        case = audit_query(group, catalog(pc), [("wrong", "CCO")], rules, confidence=.99)
        self.assertTrue(case["protected"])
        self.assertEqual(case["chemical_audit"]["strict_rule_matches"], 1)
        self.assertIn("phosphocholine_184_positive", case["chemical_audit"]["truth_supported_rule_ids"])
        bad = group.copy()
        bad["ionization_mode"] = "Positive"
        audit = rule_gate_audit(bad, rules, pc)
        self.assertEqual(audit["strict_rule_matches"], 0)
        self.assertEqual(audit["alias_restored_rule_matches"], 1)
        self.assertTrue(audit["mode_or_whitespace_encoding_issue_observed"])
        self.assertEqual(bad.iloc[0].ionization_mode, "Positive")

    def test_weak_peak_and_adduct_gates_are_distinguishable(self):
        rules = load_rules(Path(__file__).resolve().parents[1] / "configs/chemical_priors.json")
        pc = "C[N+](C)(C)CCOP(=O)([O-])OCC"
        group = query(pc, peaks=[184.0733209, 50.], intensity=[.001, 1.])
        audit = rule_gate_audit(group, rules, pc)
        gate = next(g for g in audit["rule_gates"] if g["rule_id"] == "phosphocholine_184_positive")
        self.assertEqual(gate["weak_only_joint_eligible"], 1)
        self.assertEqual(audit["strict_rule_matches"], 0)
        group["adduct"] = "[M+Na]+"
        group.at[0, "ms2_normalized_intensities"] = [1., 1.]
        gate = next(g for g in rule_gate_audit(group, rules)["rule_gates"] if g["rule_id"] == "phosphocholine_184_positive")
        self.assertEqual(gate["joint_eligible"], 0)
        self.assertEqual(gate["mass_match_ignoring_mode_adduct"], 1)

    def test_split_metrics_and_representative_cases_are_serializable(self):
        first = audit_query(query(), catalog("CCO"), [("truth", "CCO")], ())
        second = audit_query(query(), catalog("CCO"), [], ())
        first["split"], second["split"] = "train", "accept"
        summary = summarize_cases([first, second])
        self.assertEqual(summary["by_split"]["train"]["mrr25"], 1.)
        self.assertEqual(summary["by_split"]["accept"]["mrr25"], 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            report = save_audit(path, [first, second], metadata={"independent": False})
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertEqual(len(report["representative_cases"]), 1)

    def test_exported_outputs_do_not_invent_missing_mass_or_metadata(self):
        score = {"key": "truth", "truth_identity": "truth", "reciprocal_rank": .25}
        behavior = {"groups": {
            "A_original_no_rules": {"identity_results": [score], "audit": [{"molecule_id": "truth", "confidence": .8, "protected": True}]},
            "B_expanded_no_rules": {"identity_results": [{**score, "reciprocal_rank": .2}]},
        }}
        report = summarize_behavior(behavior)
        self.assertEqual(report["protected_wrong_top1"], 1)
        self.assertEqual(report["expanded_rr_losses"][0]["old_rank"], 4)
        self.assertEqual(report["expanded_rr_losses"][0]["expanded_rank"], 5)
        self.assertNotIn("truth_mass_eligible", report["cases"][0])
        self.assertEqual(report["unprotected_molecules_actually_eligible_for_rule_extraction"], 0)

    def test_existing_identity_column_avoids_mutating_or_recanonicalizing_source(self):
        source = pd.DataFrame([{"normalized_smiles": "source-text", "mass": 46., "origin": "library", "identity": "already-frozen-key"}])
        prepared = prepare_catalog(source)
        self.assertEqual(prepared.iloc[0].identity, "already-frozen-key")
        self.assertEqual(source.iloc[0].normalized_smiles, "source-text")
        with self.assertRaisesRegex(ValueError, "invalid SMILES"):
            canonical_identity("invalid")


if __name__ == "__main__":
    unittest.main()
