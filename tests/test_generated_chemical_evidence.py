import unittest

from casmi_ml.chemistry import extract_evidence, score_candidate
from casmi_ml.generated_chemical_evidence import evidence_order


class ChemicalGeneratedRankingTests(unittest.TestCase):
    def test_loss_evidence_changes_novel_order_and_excludes_retrieval(self):
        evidence = extract_evidence(
            {
                "ms2_mzs": [100 - 18.010564684],
                "ms2_normalized_intensities": [1.0],
                "precursor_mz": 100,
                "adduct": "[M+H]+",
                "instrument_type": "Orbitrap",
                "ionization_mode": "positive",
            }
        )
        scores = {
            "hydrocarbon": score_candidate([evidence], "CCC"),
            "alcohol": score_candidate([evidence], "CCO"),
            "reference": score_candidate([evidence], "O"),
        }
        self.assertGreater(scores["alcohol"]["loss"], scores["hydrocarbon"]["loss"])
        self.assertEqual(
            evidence_order(
                ["reference", "hydrocarbon", "alcohol"],
                ["reference"],
                scores,
                "loss",
                0,
            ),
            ["hydrocarbon", "alcohol"],
        )
        self.assertEqual(
            evidence_order(
                ["reference", "hydrocarbon", "alcohol"],
                ["reference"],
                scores,
                "loss",
                1.0,
            ),
            ["alcohol", "hydrocarbon"],
        )
        self.assertEqual(
            score_candidate([evidence, evidence], "CCO"), scores["alcohol"]
        )

    def test_unsupported_instrument_and_tied_scores_preserve_order(self):
        evidence = extract_evidence(
            {
                "ms2_mzs": [81.989435316],
                "ms2_normalized_intensities": [1.0],
                "precursor_mz": 100,
                "adduct": "[M+H]+",
                "instrument_type": "unknown",
                "ionization_mode": "positive",
            }
        )
        scores = {
            k: score_candidate([evidence], smiles)
            for k, smiles in [("a", "CCC"), ("b", "CCO")]
        }
        for component in ("diagnostic", "loss", "combined"):
            self.assertEqual(
                evidence_order(["a", "b"], [], scores, component, 0.5), ["a", "b"]
            )
