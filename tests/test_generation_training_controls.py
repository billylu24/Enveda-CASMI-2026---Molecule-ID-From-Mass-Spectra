import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.generation_condition_training import main, train


class TrainingControlTests(unittest.TestCase):
    def test_invalid_learning_rate_rejected_before_assets_or_output(self):
        for rate in [0, -1, float("nan"), 0.01]:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                with self.assertRaises(ValueError):
                    train(output, "single", learning_rate=rate)
                self.assertFalse(output.exists())

    def test_cli_forwards_explicit_learning_rate_and_initial_checkpoint(self):
        with (
            patch(
                "sys.argv",
                [
                    "training",
                    "--output",
                    "run",
                    "--averaging",
                    "single",
                    "--learning-rate",
                    "0.0001",
                    "--initial-checkpoint",
                    "selected.pt",
                ],
            ),
            patch(
                "casmi_ml.generation_condition_training.train", return_value={}
            ) as call,
        ):
            main()
        self.assertEqual(call.call_args.args[-2:], (0.0001, Path("selected.pt")))
