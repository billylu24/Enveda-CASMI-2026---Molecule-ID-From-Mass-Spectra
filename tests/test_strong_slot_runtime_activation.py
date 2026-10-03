import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import release_identity
from casmi_ml.strong_slot_runtime_activation import validate


class StrongSlotRuntimeGateTests(unittest.TestCase):
    def fixture(self, root):
        old, new = root / "old", root / "new"
        for p in [old, new]:
            recipe = {
                "external_routed": {
                    "high_fragment": {
                        "strong_slots": {
                            "reference_threshold": 0.5,
                            "original_prefix": 3,
                            "slots": 3,
                        }
                    }
                }
            }
            if p == new:
                recipe["reference_runtime"] = {"scan": "shared_union_v1"}
            write_json(p / "bundle/deployment_recipe.json", recipe)
            (p / "bundle/model.pt").write_bytes(b"fixed")
            sums = {
                str(f.relative_to(p / "bundle")): digest(f)
                for f in (p / "bundle").iterdir()
            }
            write_json(p / "bundle/SHA256SUMS.json", sums)
            write_json(p / "notebook/kernel-metadata.json", {"id": "owner/" + p.name})
            write_json(
                p / "verification.json",
                {
                    "valid": True,
                    "identity": release_identity(p, sums, "strong_slot_prefix"),
                    "molecules": 400,
                    "seconds": 100,
                    "peak_rss_mib": 100,
                },
            )
            (p / "local_output").mkdir()
            pd.DataFrame({"molecule_id": range(400), "smiles": ["CCO"] * 400}).to_csv(
                p / "local_output/submission.csv", index=False
            )
        write_json(
            new / "implementation_replay.json",
            {
                "valid": True,
                "full_rank_matches": 75,
                "fresh_fragment_cache": True,
                "bundle_sums_sha256": digest(new / "bundle/SHA256SUMS.json"),
            },
        )
        return old, new, {"inference_seconds": 1800, "inference_rss_mib": 8192}

    def test_only_identical400_and_bound_evidence_accepts_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, config = self.fixture(Path(directory))
            self.assertEqual(validate(old, new, config)["full_rank_matches"], 400)
            data = pd.read_csv(new / "local_output/submission.csv")
            data.loc[399, "smiles"] = "CCN"
            data.to_csv(new / "local_output/submission.csv", index=False)
            with self.assertRaisesRegex(ValueError, "All400"):
                validate(old, new, config)

    def test_changed_model_and_stale_replay_and_resource_overrun_rejected(self):
        for cause in ["model", "replay", "resource"]:
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                old, new, config = self.fixture(Path(directory))
                if cause == "model":
                    (new / "bundle/model.pt").write_bytes(b"changed")
                elif cause == "replay":
                    p = new / "implementation_replay.json"
                    x = json.loads(p.read_text())
                    x["bundle_sums_sha256"] = "wrong"
                    write_json(p, x)
                else:
                    p = new / "verification.json"
                    x = json.loads(p.read_text())
                    x["seconds"] = 1801
                    write_json(p, x)
                with self.assertRaises(ValueError):
                    validate(old, new, config)
