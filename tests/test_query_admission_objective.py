import unittest

import torch
from torch.nn import functional as F

from casmi_ml.query_admission_objective import admission_logits


class QueryAdmissionObjectiveTests(unittest.TestCase):
    def test_margin_is_in_cosine_units_before_frozen_scale_and_inputs_untouched(self):
        scores = torch.tensor([[0.4, 0.38] + [-0.5] * 14], requires_grad=True)
        before = scores.detach().clone()
        adjusted = admission_logits(scores, 25.0)
        torch.testing.assert_close(adjusted[:, 1:], scores[:, 1:] * 25, rtol=0, atol=0)
        self.assertAlmostEqual(
            float(adjusted[0, 0].detach()), (0.4 - 0.05) * 25, places=5
        )
        torch.testing.assert_close(scores.detach(), before, rtol=0, atol=0)
        self.assertLess(float(adjusted[0, 0].detach()), float(adjusted[0, 1].detach()))
        loss = F.cross_entropy(adjusted, torch.zeros(1, dtype=torch.long))
        loss.backward()
        self.assertLess(float(scores.grad[0, 0]), 0)
        self.assertGreater(float(scores.grad[0, 1]), 0)
        with self.assertRaisesRegex(ValueError, "15 pinned"):
            admission_logits(torch.zeros(3, 15), 25)
        with self.assertRaisesRegex(ValueError, "Finite"):
            admission_logits(torch.full((3, 16), float("nan")), 25)
