import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from casmi_ml.generation_experiment import generate


class GenerationTemperatureTests(unittest.TestCase):
    def test_temperature_forwarding_and_cache_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "generation/smiles_42/model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            (root / "researchdev.parquet").write_bytes(b"split")
            np.save(root / "condition.npy", np.zeros((1, 2), dtype=np.float32))
            decoder = MagicMock()
            decoder.generate.return_value = (
                torch.tensor([[1, 2]]),
                torch.tensor([0.0]),
                torch.tensor([True]),
            )
            formula = MagicMock()
            formula.hypotheses.return_value = []
            formula.soft.return_value = torch.zeros((1, 2))
            frame = pd.DataFrame(
                {"inchikey14": ["A"], "adduct": ["[M+H]+"], "precursor_mz": [101.0]}
            )
            with (
                patch(
                    "casmi_ml.generation_experiment.load_model",
                    return_value=(decoder, formula, []),
                ),
                patch(
                    "casmi_ml.generation_experiment.pd.read_parquet", return_value=frame
                ),
                patch("casmi_ml.generation_experiment.conditions", return_value=root),
                patch(
                    "casmi_ml.generation_experiment.extract_evidence", return_value={}
                ),
                patch(
                    "casmi_ml.generation_experiment.validate_generated",
                    return_value=([], {"samples": 1}),
                ),
                patch(
                    "casmi_ml.generation_experiment.torch.cuda.is_available",
                    return_value=False,
                ),
            ):
                for temperature in [0.8, 0.7, 0.9, 0.8]:
                    generate(root, samples=1, temperature=temperature)
                self.assertEqual(
                    [c.kwargs["temperature"] for c in decoder.generate.call_args_list],
                    [0.8, 0.7, 0.9],
                )
            caches = sorted(
                p.name
                for p in (root / "generation").glob("*.json")
                if ".config." not in p.name
            )
            self.assertEqual(len(caches), 3)
            self.assertIn("researchdev_samples1_limitall.json", caches)
            self.assertTrue(any("temperature0.7" in p for p in caches))
            self.assertTrue(any("temperature0.9" in p for p in caches))

    def test_nonfinite_temperature_fails_before_loading_assets(self):
        for temperature in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                generate("missing", temperature=temperature)


if __name__ == "__main__":
    unittest.main()
