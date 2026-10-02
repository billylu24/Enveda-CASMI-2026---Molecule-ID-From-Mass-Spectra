import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import PROTON
from casmi_ml.chemistry import (
    clean_peaks,
    composition_mass,
    extract_evidence,
    neutral_mass,
    parse_adduct,
    rerank,
    score_candidate,
)
from casmi_ml.research_protocol import choose_keys, freeze


def spectrum(
    peaks, precursor=150.0, adduct="[M+H]+", instrument="LC-ESI-QTOF", intensity=None
):
    return {
        "ms2_mzs": peaks,
        "ms2_normalized_intensities": intensity or [1.0] * len(peaks),
        "precursor_mz": precursor,
        "adduct": adduct,
        "instrument_type": instrument,
        "ionization_mode": "negative" if adduct.endswith("-") else "positive",
    }


class ChemicalEvidenceTests(unittest.TestCase):
    def test_charge_multiplicity_and_electron_corrections(self):
        for adduct in [
            "[M+H]+",
            "[M-H]-",
            "[M+2H]2+",
            "[M+3H]3+",
            "[2M+Na]+",
            "[2M-H]-",
            "[M]+",
        ]:
            ion = parse_adduct(adduct)
            mz = (300.0 * ion.molecules + ion.shift) / abs(ion.charge)
            self.assertAlmostEqual(
                neutral_mass({"adduct": adduct, "precursor_mz": mz}), 300.0, places=8
            )
        self.assertAlmostEqual(parse_adduct("[M+H]+").shift, PROTON, places=7)
        self.assertIsNone(parse_adduct("[Cat]2+"))
        self.assertIsNone(parse_adduct("[M]0+"))
        self.assertIsNone(neutral_mass({"adduct": "unknown", "precursor_mz": 123}))

    def test_carbon_monoxide_is_not_nitrogen_loss(self):
        water = composition_mass("H2O")
        co = composition_mass("CO")
        n2 = composition_mass("N2")
        e = extract_evidence(
            spectrum([150 - co, 150 - n2, 150 - water], intensity=[1.0, 0.8, 0.001])
        )
        hits = [
            m
            for m in e["matches"]
            if m["rule"] == "carbon_monoxide" and m["kind"] == "precursor_loss"
        ]
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["mz"], 150 - co)
        self.assertTrue(any(m["rule"] == "water" for m in e["matches"]))

    def test_unknown_resolution_and_multicharge_do_not_assert_losses(self):
        for instrument, adduct in [
            ("Ion Trap", "[M+H]+"),
            ("QTOF", "[M+2H]2+"),
            ("QTOF", "[2M+H]+"),
            (None, "[M+H]+"),
        ]:
            e = extract_evidence(
                spectrum(
                    [150 - composition_mass("H2O")],
                    adduct=adduct,
                    instrument=instrument,
                )
            )
            self.assertFalse(e["matches"])
            self.assertTrue(e["observations"])

    def test_continuous_losses_and_non_specific_support(self):
        water = composition_mass("H2O")
        e = extract_evidence(spectrum([150 - water, 150 - 2 * water]))
        self.assertTrue(
            any(c["rules"] == ["water", "water"] for c in e["combinations"])
        )
        score = score_candidate([e], "CCO")
        self.assertGreater(score["loss"], 0)
        self.assertGreater(score["combination"], 0)
        self.assertEqual(score_candidate([e, e], "CCO"), score)
        self.assertEqual(score_candidate([e], "CCC")["score"], 0)

    def test_phosphocholine_requires_mode_and_substructure(self):
        mz = composition_mass("C5H15NO4P") - 0.000548579909
        e = extract_evidence(spectrum([mz], precursor=600))
        self.assertTrue(any(m["rule"] == "phosphocholine" for m in e["matches"]))
        smi = "COP(=O)(O)OCC[N+](C)(C)C"
        self.assertGreater(score_candidate([e], smi)["diagnostic"], 0)
        negative = extract_evidence(spectrum([mz], precursor=600, adduct="[M-H]-"))
        self.assertFalse(
            any(m["rule"] == "phosphocholine" for m in negative["matches"])
        )

    def test_absence_ties_and_empty_spectra_are_neutral(self):
        e = extract_evidence(spectrum([]))
        self.assertEqual(score_candidate([e], "CCO")["score"], 0)
        self.assertEqual(rerank(["b", "a"], {"b": "CCC", "a": "CCO"}, [e]), ["b", "a"])
        self.assertEqual(
            rerank(["b", "a"], {}, [], fragment_scores={"b": 3.0, "a": 3.0}), ["b", "a"]
        )
        self.assertFalse(score_candidate([e], "not_a_smiles")["valid"])
        self.assertEqual(rerank(["a", "b"], {}, [], weight=0), ["a", "b"])

    def test_peak_validation(self):
        with self.assertRaises(ValueError):
            clean_peaks({"ms2_mzs": [1], "ms2_normalized_intensities": []})
        mz, i = clean_peaks(
            {"ms2_mzs": [np.nan, 30, 20], "ms2_normalized_intensities": [1, -1, 0.5]}
        )
        np.testing.assert_equal(mz, [20])
        np.testing.assert_equal(i, [1])

    def test_cohort_exclusion_and_frozen_protocol(self):
        cat = pd.DataFrame(
            {
                "inchikey14": ["a", "b", "c", "d"],
                "split": ["dev", "dev", "holdout", "dev"],
            }
        )
        selected = choose_keys(cat, {"a"}, 2, "dev", 42)
        self.assertEqual(set(selected), {"b", "d"})
        self.assertEqual(selected, choose_keys(cat, {"a"}, 2, "dev", 42))
        with self.assertRaises(ValueError):
            choose_keys(cat, {"a"}, 3, "dev", 42)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "protocol.json"
            freeze(p, {"version": 1})
            with self.assertRaises(ValueError):
                freeze(p, {"version": 2})


if __name__ == "__main__":
    unittest.main()


class FragmentBudgetTests(unittest.TestCase):
    def test_expired_deadline_skips_java(self):
        from unittest.mock import Mock, patch

        from casmi_ml.metfrag import score_group

        tool = Mock(timeout=60)
        with patch("casmi_ml.metfrag.time.monotonic", return_value=10):
            self.assertEqual(score_group(tool, [{}], {"a": "CCO"}, 9), ({}, True))
        tool.score.assert_not_called()
        self.assertEqual(tool.timeout, 60)

    def test_deadline_uses_remaining_timeout_then_discards_partial_group(self):
        from unittest.mock import Mock, patch

        from casmi_ml.metfrag import score_group

        tool = Mock(timeout=60)

        def score(row, candidates):
            self.assertEqual(tool.timeout, 2)
            return {"scores": {"a": 1.0}}

        tool.score.side_effect = score
        with patch("casmi_ml.metfrag.time.monotonic", side_effect=[8, 10]):
            self.assertEqual(score_group(tool, [{}], {"a": "CCO"}, 10), ({}, True))
        self.assertEqual(tool.timeout, 60)

    def test_unbounded_group_keeps_maximum_evidence_across_spectra(self):
        from unittest.mock import Mock

        from casmi_ml.metfrag import score_group

        tool = Mock(timeout=60)
        tool.score.side_effect = [
            {"scores": {"a": 4.0}},
            {"scores": {"a": 2.0, "b": 3.0}},
        ]
        self.assertEqual(score_group(tool, [{}, {}], {}), ({"a": 4.0, "b": 3.0}, False))
        self.assertEqual(tool.timeout, 60)
