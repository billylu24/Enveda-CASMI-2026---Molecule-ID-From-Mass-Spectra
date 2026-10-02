import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.critic_hard_negative_training import main


class CriticTrainingCLITests(unittest.TestCase):
    def test_fixed_epoch_and_hard_arm_reach_training(self):
        with (
            patch("sys.argv", ["train", "--output", "out", "--hard", "--epochs", "6"]),
            patch("casmi_ml.critic_hard_negative_training.run", return_value={}) as run,
            patch("builtins.print"),
        ):
            main()
        run.assert_called_once_with(Path("out"), True, 6)
