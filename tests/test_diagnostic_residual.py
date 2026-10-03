import unittest

import numpy as np
import torch
from torch import nn

from casmi_ml.diagnostic_residual import DIM, DiagnosticResidual, evidence_features
from tests.test_chemistry import spectrum


class DiagnosticResidualTests(unittest.TestCase):
    def test_fine_mass_discriminates_co_from_nitrogen_in_same_integer_bin(self):
        nitrogen = spectrum([150 - 28.006148])
        fine = evidence_features(nitrogen)
        coarse = evidence_features(nitrogen, False)
        self.assertLess(float(fine.sum()), float(coarse.sum()))
        changed = dict(
            nitrogen,
            inchikey14="label",
            normalized_smiles="CCO",
            molecular_formula="C2H6O",
        )
        np.testing.assert_array_equal(fine, evidence_features(changed))

    def test_zero_evidence_and_zero_initialized_residual_preserve_base_exactly(self):
        base = nn.Linear(2, 2048)
        model = DiagnosticResidual(base)
        for chemical in [torch.zeros(4, DIM), torch.ones(4, DIM)]:
            torch.testing.assert_close(
                model.residual(chemical), torch.zeros(4, 2048), rtol=0, atol=0
            )
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        model.train()
        self.assertFalse(base.training)

    def test_peak_to_peak_loss_has_a_separate_nonzero_channel(self):
        row = spectrum([100.0, 127.994915], precursor=300.0)
        values = evidence_features(row)
        self.assertGreater(values[2 * 3 + 2], 0)

    def test_loader_and_group_prediction_use_explicit_evidence_features(self):
        import tempfile
        from pathlib import Path

        import pandas as pd

        from casmi_ml.data import fit_preprocessing
        from casmi_ml.inference import group_probability
        from casmi_ml.scale_models import ScaleModel
        from casmi_ml.secondary_inference import load_deployment_checkpoint

        group = pd.DataFrame([spectrum([131.989435])])
        group["collision_energy_ev"] = [[]]
        prep = fit_preprocessing(group)
        dimension = 6 + sum(len(x) + 1 for x in prep["categories"].values())
        base = ScaleModel("wide_enhanced", dimension).eval()
        model = DiagnosticResidual(base).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {
                    "architecture": "wide_enhanced",
                    "metadata_dim": dimension,
                    "state_dict": model.state_dict(),
                    "diagnostic_residual": {"precise": True},
                    "preprocessing": dict(prep, diagnostic_precision=True),
                },
                path,
            )
            loaded, saved = load_deployment_checkpoint(path, "scale")
            with torch.inference_mode():
                original = group_probability(base, group, prep)
                residual = group_probability(loaded, group, saved["preprocessing"])
            np.testing.assert_array_equal(original, residual)
