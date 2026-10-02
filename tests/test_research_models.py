import unittest

import numpy as np
import torch
from rdkit import Chem

from casmi_ml.generation_experiment import (
    FormulaPredictor,
    formula_counts,
    validate_generated,
)
from casmi_ml.research_models import (
    Distillation,
    PeakEncoder,
    SmilesDecoder,
    SmilesVocabulary,
    augmented,
    latent_diagnostics,
    masked_loss,
    peak_tokens,
)


class ResearchModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.batch = {
            "peaks": torch.randn(2, 8, 19),
            "mask": torch.ones(2, 8, dtype=torch.bool),
            "protected": torch.zeros(2, 8, dtype=torch.bool),
            "meta": torch.randn(2, 5),
        }

    def test_encoder_permutation_invariance(self):
        model = PeakEncoder(5, width=32, layers=1).eval()
        order = torch.randperm(8)
        other = {
            k: v[:, order] if k in ["peaks", "mask", "protected"] else v
            for k, v in self.batch.items()
        }
        with torch.no_grad():
            torch.testing.assert_close(
                model(self.batch), model(other), atol=1e-5, rtol=1e-5
            )

    def test_teacher_has_no_grad_and_ema_updates(self):
        model = Distillation(PeakEncoder(5, width=32, layers=1), prototypes=16).train()
        before = [p.detach().clone() for p in model.teacher.parameters()]
        loss = model.loss(self.batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(model.teacher.training)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        with torch.no_grad():
            next(model.student.parameters()).add_(1)
        model.update_teacher(0.5)
        self.assertFalse(torch.equal(before[0], next(model.teacher.parameters())))

    def test_views_keep_diagnostics_and_at_least_one_peak(self):
        self.batch["peaks"][..., 1] = 0.1
        self.batch["protected"][:, 3] = True
        view = augmented(self.batch, drop=1.0)
        self.assertTrue(view["mask"][:, 3].all())
        self.assertTrue(view["mask"].any(1).all())
        torch.testing.assert_close(view["peaks"][..., 0], self.batch["peaks"][..., 0])
        self.batch["protected"][:] = False
        self.assertTrue(augmented(self.batch, drop=1.0)["mask"].any(1).all())

    def test_weak_diagnostic_peak_survives_token_selection(self):
        row = {
            "ms2_mzs": list(range(10, 200)) + [184.0733],
            "ms2_normalized_intensities": [1.0] * 190 + [0.0001],
            "precursor_mz": 600.0,
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
            "instrument_type": "QTOF",
        }
        tokens, mask, protected = peak_tokens(row)
        self.assertTrue(protected.any())
        self.assertTrue(np.any(abs(tokens[protected, 0] * 1250 - 184.0733) < 0.002))
        self.assertEqual(mask.sum(), 128)
        tokens, mask, protected = peak_tokens(
            {"ms2_mzs": [], "ms2_normalized_intensities": []}
        )
        self.assertEqual(mask.sum(), 1)

    def test_masking_removes_mass_and_loss_target_leakage(self):
        model = PeakEncoder(5, width=32, layers=1)
        captured = {}

        def hook(module, args):
            captured["tokens"] = args[0].clone()

        h = model.embedding.register_forward_pre_hook(hook)
        loss = masked_loss(model, self.batch)
        h.remove()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((captured["tokens"] == 0).all(-1).any())

    def test_collapse_detection(self):
        self.assertTrue(latent_diagnostics(np.ones((20, 8)))["collapsed"])
        self.assertFalse(
            latent_diagnostics(np.random.default_rng(42).normal(size=(20, 8)))[
                "collapsed"
            ]
        )

    def test_decoder_is_causal(self):
        model = SmilesDecoder(12, 5, width=32, layers=1, limit=16).eval()
        condition = torch.rand(2, 5)
        a = torch.tensor([[1, 4, 5, 6], [1, 5, 6, 7]])
        b = a.clone()
        b[:, 2:] = 9
        with torch.no_grad():
            torch.testing.assert_close(
                model(a, condition)[:, :2], model(b, condition)[:, :2]
            )

    def test_vocabulary_and_generated_candidates_no_oracle(self):
        v = SmilesVocabulary.fit(["CCO", "COC", "C[N+](C)(C)C", "ClCBr"])
        for s in ["CCO", "COC", "ClCBr"]:
            self.assertEqual(v.decode(v.encode(s)), s)
        self.assertIsNone(v.encode("C" * 300))
        sequences = [v.encode("CCO"), v.encode("CCO"), v.encode("COC")]
        candidates, stats = validate_generated(
            sequences, [-2, -3, -2], [True, True, False], v, 46.0418648, [], []
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(stats["mass_matching"], 2)
        self.assertEqual(
            candidates[0]["key"], Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))[:14]
        )
        self.assertEqual(
            validate_generated(sequences, [-2] * 3, [True] * 3, v, 100.0, [], [])[0], []
        )
        self.assertEqual(
            validate_generated(sequences, [-2] * 3, [True] * 3, v, None, [], [])[0], []
        )

    def test_formula_hypotheses_are_predicted_and_valid_shapes(self):
        model = FormulaPredictor(5)
        z = torch.randn(1, 5)
        self.assertEqual(model.soft(z).shape, (1, 14))
        hypotheses = model.hypotheses(z, k=5)
        self.assertEqual(len(hypotheses), 5)
        self.assertIsNotNone(formula_counts("C2H6O"))
        self.assertIsNone(formula_counts("bad_formula"))


if __name__ == "__main__":
    unittest.main()


class GenerationDeadlineTests(unittest.TestCase):
    def test_deadline_interrupts_before_decoding_partial_candidates(self):
        import time

        import torch

        from casmi_ml.research_models import SmilesDecoder

        decoder = SmilesDecoder(8, 4, width=16, layers=1)
        with self.assertRaises(TimeoutError):
            decoder.generate(
                torch.zeros(1, 4), samples=2, deadline=time.monotonic() - 1
            )


class AtomMassConstraintTests(unittest.TestCase):
    def test_token_lower_bound_handles_isotopes_aromatic_and_brackets(self):
        from rdkit import Chem

        from casmi_ml.generation_constraints import token_atom_masses
        from casmi_ml.research_models import SmilesVocabulary

        vocabulary = SmilesVocabulary(["C", "c", "[nH]", "[81Br]", "Cl", "(", "1"])
        masses = dict(zip(vocabulary.tokens, token_atom_masses(vocabulary)))
        table = Chem.GetPeriodicTable()
        self.assertEqual(masses["C"], masses["c"])
        self.assertEqual(masses["[nH]"], table.GetMostCommonIsotopeMass(7))
        self.assertEqual(masses["[81Br]"], table.GetMassForIsotope(35, 81))
        for token in ["<eos>", "(", "1"]:
            self.assertEqual(masses[token], 0)

    def test_sampler_masks_oversized_atoms_and_preserves_unconstrained_path(self):
        from unittest.mock import patch

        import torch

        from casmi_ml.generation_constraints import token_atom_masses
        from casmi_ml.research_models import SmilesDecoder, SmilesVocabulary

        vocabulary = SmilesVocabulary(["C", "Br"])
        decoder = SmilesDecoder(len(vocabulary.tokens), 4, width=32, layers=1, limit=3)

        def logits(tokens, condition):
            values = torch.full(
                (len(tokens), tokens.shape[1], len(vocabulary.tokens)), -100.0
            )
            values[:, :, vocabulary.ids["Br"]] = 30.0
            values[:, :, vocabulary.ids["C"]] = 20.0
            values[:, :, 2] = 0.0
            return values

        with patch.object(decoder, "forward", side_effect=logits):
            raw = decoder.generate(
                torch.zeros(1, 4),
                samples=1,
                generator=torch.Generator().manual_seed(42),
            )
            constrained = decoder.generate(
                torch.zeros(1, 4),
                samples=1,
                generator=torch.Generator().manual_seed(42),
                token_masses=token_atom_masses(vocabulary),
                neutral_mass=16.04,
            )
        self.assertEqual(raw[0][0, 1].item(), vocabulary.ids["Br"])
        self.assertEqual(constrained[0].tolist(), [[1, vocabulary.ids["C"], 2]])
        self.assertTrue(constrained[2][0])
        with self.assertRaises(ValueError):
            decoder.generate(
                torch.zeros(1, 4), token_masses=token_atom_masses(vocabulary)
            )
