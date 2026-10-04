import unittest

import numpy as np

from casmi_ml.chemistry import composition_mass
from casmi_ml.generation_experiment import ELEMENTS, formula_counts
from casmi_ml.mass_formula_proposal_audit import joint_beam, mass_proposals


class MassFormulaProposalTest(unittest.TestCase):
    def test_observed_mass_keeps_lower_probability_compatible_formula(self):
        formulas = sorted(["CH4", "C2H6", "C2H6O"], key=composition_mass)
        counts = np.stack([formula_counts(f) for f in formulas])
        masses = np.asarray([composition_mass(f) for f in formulas])
        probabilities = np.full((len(ELEMENTS), 257), -100.0)
        for i, value in enumerate(formula_counts("CH4")):
            probabilities[i, value] = 0.0
        self.assertEqual(joint_beam(probabilities, k=1), ["CH4"])
        proposal, n = mass_proposals(
            probabilities, composition_mass("C2H6O"), masses, counts, formulas
        )
        self.assertEqual((proposal, n), (["C2H6O"], 1))

    def test_mass_only_control_uses_the_same_candidate_window(self):
        formulas = ["C2H6O", "CH4O2"]
        masses = np.array([100.0, 100.0001])
        counts = np.stack([formula_counts(f) for f in formulas])
        logits = np.zeros((len(ELEMENTS), 257))
        neural, n = mass_proposals(logits, 100, masses, counts, formulas, k=1)
        prior, other_n = mass_proposals(
            logits,
            100,
            masses,
            counts,
            formulas,
            k=1,
            formula_log_prior=np.log([0.1, 0.9]),
        )
        self.assertEqual(n, other_n)
        self.assertEqual(n, 2)
        self.assertEqual(neural, ["C2H6O"])
        self.assertEqual(prior, ["CH4O2"])

    def test_no_candidate_and_missing_mass_keep_empty_proposals(self):
        formulas = ["CH4"]
        counts = np.stack([formula_counts("CH4")])
        masses = np.asarray([composition_mass("CH4")])
        logits = np.zeros((len(ELEMENTS), 257))
        for center in (None, float("nan"), -1, 200):
            self.assertEqual(
                mass_proposals(logits, center, masses, counts, formulas), ([], 0)
            )

    def test_equal_scores_are_deterministic_and_do_not_inject_absent_formula(self):
        formulas = ["C2H6O", "C2H6O"]
        counts = np.stack([formula_counts(f) for f in formulas])
        masses = np.asarray([composition_mass(f) for f in formulas])
        proposals, n = mass_proposals(
            np.zeros((len(ELEMENTS), 257)), masses[0], masses, counts, formulas, k=1
        )
        self.assertEqual((proposals, n), (["C2H6O"], 2))
