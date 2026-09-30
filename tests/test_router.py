import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml import router_experiment
from casmi_ml.router_experiment import FEATURES, chosen_actions, policy_configs, score


class RouterTests(unittest.TestCase):
    def test_encoder_cache_is_separate_from_control_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "fresh_probabilities.npy", np.array([0.1]))
            np.save(root / "encoder_fresh_probabilities.npy", np.array([0.9]))
            with patch.object(router_experiment, "ROOT", root):
                np.testing.assert_array_equal(
                    router_experiment.prepare_probabilities("fresh"), [0.9]
                )

    def test_only_observable_features_and_predeclared_policies(self):
        self.assertFalse(
            set(FEATURES) & {"key", "mode", "known", "covered", "label", "truth"}
        )
        configs = policy_configs()
        self.assertEqual(len(configs), 53)
        self.assertEqual(len({c["name"] for c in configs}), 53)

    def test_routing_and_score_use_correct_actions(self):
        rows = [
            {
                "key": "a",
                "mode": "unknown",
                "features": [0.4],
                "baseline_action": "F",
                "rr": {"H": 0.0, "F": 0.5, "U": 0.2, "N": 1.0},
                "covered": True,
            },
            {
                "key": "b",
                "mode": "known",
                "features": [0.9],
                "baseline_action": "H",
                "rr": {"H": 1.0, "F": 0.25, "U": 1.0, "N": 0.0},
                "covered": True,
            },
        ]
        fixed = {"type": "fixed", "threshold": 0.5, "high": "H", "low": "F"}
        self.assertEqual(chosen_actions(rows, fixed), ["F", "H"])
        learned = {"type": "boosted", "threshold": 0.4, "high": "U", "low": "N"}
        actions = chosen_actions(rows, learned, np.asarray([0.1, 0.8]))
        self.assertEqual(actions, ["N", "U"])
        for mode in ["known", "unknown"]:
            report, _ = score(rows, actions, mode)
            self.assertEqual(report["molecules"], 1)
            self.assertEqual(report["mrr25"], 1.0)
            self.assertEqual(report["top1"], 1.0)


if __name__ == "__main__":
    unittest.main()
