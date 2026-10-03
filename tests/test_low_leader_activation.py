import copy
import json
import unittest
from pathlib import Path

from casmi_ml.low_leader_activation import validate_replay
from casmi_ml.metfrag import digest


class LowLeaderEvidenceGateTests(unittest.TestCase):
    def fixture(self):
        config = json.loads(Path("configs/research_loop.json").read_text())
        report = {
            "verified": True,
            "local_only": True,
            "development_budget_fallbacks": 0,
            "incremental_new_failure_contexts": 0,
        }
        baseline = {}
        for mode, count, mrr in [("unknown", 2000, 0.04), ("known", 1340, 0.60)]:
            baseline[mode] = {"molecules": count, "mrr25": mrr, "top1": 0.5}
            report[mode] = {
                "molecules": count,
                "full_rank_matches": count,
                "metrics": {
                    **baseline[mode],
                    "mrr25": mrr + 0.002 if mode == "unknown" else mrr - 0.0008,
                },
            }
        protocol = {
            "inference_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "low_leader_source_sha256": digest("casmi_ml/low_leader_inference.py"),
        }
        return report, protocol, config, baseline

    def test_matching_full_replay_and_both_gates_are_required(self):
        report, protocol, config, baseline = self.fixture()
        selected, gates = validate_replay(report, protocol, config, baseline, baseline)
        self.assertEqual(selected["unknown"]["mrr25"], 0.042)
        self.assertTrue(all(g["eligible"] for g in gates.values()))
        for field, value in [
            ("verified", False),
            ("development_budget_fallbacks", 1),
            ("incremental_new_failure_contexts", 1),
        ]:
            bad = copy.deepcopy(report)
            bad[field] = value
            with self.assertRaisesRegex(ValueError, "complete local replay"):
                validate_replay(bad, protocol, config, baseline, baseline)
        report["known"]["full_rank_matches"] -= 1
        with self.assertRaisesRegex(ValueError, "Every full ranking"):
            validate_replay(report, protocol, config, baseline, baseline)

    def test_known_regression_or_stronger_completed_control_blocks_activation(self):
        report, protocol, config, baseline = self.fixture()
        report["known"]["metrics"]["mrr25"] = 0.5989
        with self.assertRaisesRegex(ValueError, "development gates"):
            validate_replay(report, protocol, config, baseline, baseline)
        report, protocol, config, baseline = self.fixture()
        stronger = copy.deepcopy(baseline)
        stronger["unknown"]["mrr25"] = 0.0415
        with self.assertRaisesRegex(ValueError, "development gates"):
            validate_replay(report, protocol, config, baseline, stronger)
