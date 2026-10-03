import copy
import tempfile
import unittest
from pathlib import Path

import torch

from casmi_ml.diagnostic_query_ranker import DiagnosticQueryRanker, load_query_ranker
from casmi_ml.direct_models import DirectRanker
from casmi_ml.metfrag import digest


class FrozenQueryCriticTests(unittest.TestCase):
    def test_original_queries_and_molecule_scores_preserved_and_critic_is_frozen(self):
        torch.manual_seed(42)
        base = DirectRanker("fingerprint").eval()
        model = DiagnosticQueryRanker(base).train()
        latent = torch.randn(3, 768)
        chemical = torch.ones(3, 24)
        fps = torch.rand(9, 2048)
        torch.testing.assert_close(
            model.encode_spectra(latent, chemical),
            base.encode_spectra(latent),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.encode_molecules(fps), base.encode_molecules(fps), rtol=0, atol=0
        )
        self.assertFalse(base.training)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))

    def test_checkpoint_rejects_changed_original_critic_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = DirectRanker("fingerprint").eval()
            original = root / "base.pt"
            path = root / "query.pt"
            torch.save(
                {
                    "encoder_sha256": "fixed",
                    "state_dict": base.state_dict(),
                    "architecture": "fingerprint",
                },
                original,
            )
            model = DiagnosticQueryRanker(base)
            weights = {
                "encoder_sha256": "fixed",
                "state_dict": copy.deepcopy(model.state_dict()),
                "diagnostic_query_residual": {
                    "base_critic_sha256": digest(original),
                    "dimension": 24,
                    "precise": True,
                },
            }
            torch.save(weights, path)
            loaded, _ = load_query_ranker(path, original)
            self.assertIsInstance(loaded, DiagnosticQueryRanker)
            weights["state_dict"]["base.log_scale"] += 1
            torch.save(weights, path)
            with self.assertRaisesRegex(ValueError, "critic tensor changed"):
                load_query_ranker(path, original)
