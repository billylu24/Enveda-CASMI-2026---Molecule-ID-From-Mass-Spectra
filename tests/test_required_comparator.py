import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml.research_loop import Controller
from tests.test_research_loop import CONFIG, sample


class RequiredComparatorTests(unittest.TestCase):
    def evaluate(self, root, matched_mrr, corrupt=False):
        config = copy.deepcopy(CONFIG)
        config.update(root=str(root / "state"), source=str(root / "unused"))
        path = root / "config.json"
        path.write_text(json.dumps(config))
        c = Controller(path)
        base, selected, matched = sample(), sample(mrr=0.0256), sample(mrr=matched_mrr)
        directory = root / "trial"
        comparator = root / "control"
        directory.mkdir()
        comparator.mkdir()
        for mode in ("unknown", "known"):
            for variant, metric, destination in [
                ("baseline", base, directory),
                ("candidate", selected, directory),
                ("control", matched, comparator),
            ]:
                values = metric[mode]
                pd.DataFrame(
                    {
                        "key": [f"{mode}{i}" for i in range(values["molecules"])],
                        "reciprocal_rank": values["mrr25"],
                        "top1": values["top1"],
                    }
                ).to_csv(destination / f"{mode}_{variant}.csv", index=False)
        if corrupt:
            matched["unknown"]["mrr25"] += 0.001
        (directory / "protocol.json").write_text("{}")
        report = {
            mode: {"baseline": base[mode], "candidate": selected[mode]}
            for mode in ("unknown", "known")
        }
        report["required_comparators"] = {
            "complete_control": {
                "directory": str(comparator),
                "variant": "control",
                "metrics": matched,
            }
        }
        (directory / "report.json").write_text(json.dumps(report))
        c.register_round("trial", "local", [], directory / "report.json")
        with patch("casmi_ml.research_loop.write_json"):
            result = c.evaluate_round("trial")
        return result, c

    def test_historical_gain_cannot_replace_best_when_matched_gain_fails(self):
        with tempfile.TemporaryDirectory() as d:
            result, c = self.evaluate(Path(d), 0.0253)
            self.assertIsNone(result["winner"])
            candidate = next(
                x for x in result["choices"] if x["variant"] == "candidate"
            )
            self.assertIn(
                "required_comparator_failed:complete_control",
                candidate["gate"]["reasons"],
            )
            self.assertIsNone(c.read()["development_best"])

    def test_candidate_that_passes_both_controls_remains_eligible(self):
        with tempfile.TemporaryDirectory() as d:
            result, _ = self.evaluate(Path(d), 0.025)
            self.assertEqual(result["winner"]["variant"], "candidate")

    def test_comparator_report_must_match_its_paired_evidence(self):
        with (
            tempfile.TemporaryDirectory() as d,
            self.assertRaisesRegex(ValueError, "comparator differs"),
        ):
            self.evaluate(Path(d), 0.025, corrupt=True)
