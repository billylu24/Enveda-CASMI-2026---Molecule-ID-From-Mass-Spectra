"""Protect label-free inference and exact full-ranking handoffs."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_routed_inference import extend, select_high_fragment


class RoutedInferenceBoundaryTests(unittest.TestCase):
    def test_truth_columns_are_rejected_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for forbidden in (
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "fingerprint",
            ):
                path = root / "test.parquet"
                pd.DataFrame(
                    {"molecule_id": ["opaque1"], forbidden: ["label"]}
                ).to_parquet(path)
                with self.assertRaisesRegex(ValueError, "unlabeled input"):
                    extend(
                        path,
                        root / "missing.csv",
                        None,
                        None,
                        root / "output.csv",
                        root,
                        {},
                    )

    def test_current_full_ranking_must_match_baseline_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"molecule_id": ["opaque1"]}).to_parquet(root / "test.parquet")
            pd.DataFrame({"molecule_id": ["opaque1"], "smiles": ["CCO"]}).to_csv(
                root / "base.csv", index=False
            )
            (root / "full.json").write_text(
                json.dumps([{"molecule_id": "opaque1", "smiles": ["COC"]}])
            )
            with self.assertRaisesRegex(ValueError, "frozen baseline CSV"):
                extend(
                    root / "test.parquet",
                    root / "base.csv",
                    root / "full.json",
                    None,
                    root / "output.csv",
                    root,
                    {},
                )


class HighFragmentRuleTests(unittest.TestCase):
    def setUp(self):
        self.current = ["first", "a", "b", "c"]
        self.tail = self.current + ["new", "other"]
        self.candidates = ["new", "other"]
        self.critic = {"first": 0.4, "new": 0.6, "other": 0.5}

    def test_budget_fallback_keeps_existing_high_tail(self):
        ranking, _, status = select_high_fragment(
            self.current, self.tail, self.candidates, self.critic, {"new": 2.0}, True
        )
        self.assertEqual(ranking, self.tail)
        self.assertEqual(status, "high_fragment_budget_keep_tail")

    def test_missing_and_tied_fragment_evidence_keeps_existing_high_tail(self):
        for fragments in ({}, {"new": 1.0, "other": 1.0}):
            ranking, _, status = select_high_fragment(
                self.current, self.tail, self.candidates, self.critic, fragments, False
            )
            self.assertEqual(ranking, self.tail)
            self.assertEqual(status, "high_fragment_evidence_keep_tail")

    def test_actual_first_fragment_requires_strictly_better_score(self):
        ranking, inserted, status = select_high_fragment(
            self.current,
            self.tail,
            self.candidates,
            self.critic,
            {"first": 2.0, "new": 2.0, "other": 0.0},
            False,
        )
        self.assertEqual(ranking, self.current)
        self.assertEqual(inserted, [])
        self.assertEqual(status, "high_fragment_remove_tail")

    def test_supported_candidate_moves_after_original_first3(self):
        ranking, _, status = select_high_fragment(
            self.current,
            self.tail,
            self.candidates,
            self.critic,
            {"first": 1.0, "new": 2.0, "other": 0.0},
            False,
        )
        self.assertEqual(ranking[:3], self.current[:3])
        self.assertEqual(ranking[3], "new")
        self.assertEqual(status, "high_fragment_inserted")

    def test_fragment_leader_still_requires_original_critic_margin(self):
        critic = dict(self.critic, new=0.44)
        ranking, _, status = select_high_fragment(
            self.current,
            self.tail,
            self.candidates,
            critic,
            {"first": 1.0, "new": 2.0, "other": 0.0},
            False,
        )
        self.assertEqual(ranking, self.current)
        self.assertEqual(status, "high_fragment_remove_tail")
