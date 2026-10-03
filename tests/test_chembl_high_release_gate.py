import json
import tempfile
import unittest
from pathlib import Path

from casmi_ml.chembl_routed_release import prepare


class HighFragmentReleaseGateTests(unittest.TestCase):
    def test_replay_alone_does_not_allow_rejected_development_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment"
            source.mkdir()
            (source / "decision.json").write_text(
                json.dumps({"direction": "chembl_high_fragment", "winner": None})
            )
            (source / "replay.json").write_text(
                json.dumps({"valid": True, "molecules": 75})
            )
            with self.assertRaisesRegex(ValueError, "development gate"):
                prepare(source, root / "release")
            self.assertFalse((root / "release").exists())

    def test_eligible_development_still_requires_actual_unlabeled_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment"
            source.mkdir()
            (source / "decision.json").write_text(
                json.dumps(
                    {
                        "direction": "chembl_high_fragment",
                        "winner": {"gate": {"eligible": True}},
                    }
                )
            )
            (source / "replay.json").write_text(
                json.dumps({"valid": False, "molecules": 75})
            )
            with self.assertRaisesRegex(ValueError, "unlabeled external replay"):
                prepare(source, root / "release")
            self.assertFalse((root / "release").exists())

    def test181_valid_flag_with_incomplete_full_rank_match_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "experiment"
            source.mkdir()
            (source / "decision.json").write_text(
                json.dumps(
                    {
                        "direction": "chembl_high_strong_slots",
                        "winner": {"gate": {"eligible": True}},
                    }
                )
            )
            (source / "replay.json").write_text(
                json.dumps({"valid": True, "molecules": 75, "full_rank_matches": 74})
            )
            with self.assertRaisesRegex(ValueError, "source-bound181"):
                prepare(source, root / "release")
            self.assertFalse((root / "release").exists())
