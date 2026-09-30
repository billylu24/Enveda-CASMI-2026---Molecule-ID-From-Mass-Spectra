import tempfile
import unittest
from pathlib import Path

import torch

from casmi_ml.scale_models import ScaleModel
from casmi_ml.secondary_inference import load_deployment_checkpoint


class ScaleTests(unittest.TestCase):
    def test_backward_permutation_padding_and_checkpoint(self):
        torch.set_num_threads(2)
        batch = {'hist': torch.rand(2, 1250), 'loss': torch.rand(2, 1250),
                 'meta': torch.rand(2, 10), 'peaks': torch.rand(2, 64, 19),
                 'mask': torch.zeros(2, 64, dtype=torch.bool)}
        batch['mask'][:, :5] = True
        for architecture in ['metadata_control', 'wide_metadata', 'wide_enhanced',
                             'peak_transformer', 'hybrid_transformer']:
            model = ScaleModel(architecture, 10).eval()
            output = model(batch)
            self.assertEqual(output.shape, (2, 2048))
            self.assertTrue(torch.isfinite(output).all())
            output.mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if 'transformer' in architecture:
                changed = {k: v.clone() for k, v in batch.items()}
                changed['peaks'][~changed['mask']] = 100.
                torch.testing.assert_close(output, model(changed), atol=1e-5, rtol=1e-5)
                perm = torch.randperm(64)
                changed['peaks'] = batch['peaks'][:, perm]
                changed['mask'] = batch['mask'][:, perm]
                torch.testing.assert_close(output, model(changed), atol=1e-5, rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'weights.pt'
                torch.save(model.state_dict(), path)
                other = ScaleModel(architecture, 10).eval()
                other.load_state_dict(torch.load(path, weights_only=True))
                torch.testing.assert_close(output, other(batch))
                torch.save({'architecture': architecture, 'metadata_dim': 10,
                            'state_dict': model.state_dict()}, path)
                deployed, _ = load_deployment_checkpoint(path, 'scale')
                torch.testing.assert_close(output, deployed(batch))


if __name__ == '__main__':
    unittest.main()
