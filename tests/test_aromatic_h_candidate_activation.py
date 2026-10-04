import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.aromatic_h_candidate_activation import SOURCES, activate, resolve_score
from casmi_ml.metfrag import digest
from tests import test_full_gap_candidate_activation as fixtures


class AromaticHActivationTests(unittest.TestCase):
    def fixture(self, root):
        c, out, result, state = fixtures.FullGapCandidateActivationTests().fixture(root)
        protocol = json.loads((out / "protocol.json").read_text())
        protocol.update({field: digest(source) for field, source in SOURCES.items()})
        (out / "protocol.json").write_text(json.dumps(protocol))
        for rid in (
            "0348_aromatic_h_candidate_zero_control",
            "0352_aromatic_h_candidate_mask_system",
        ):
            control = root / "rounds" / rid
            control.mkdir()
            (control / "protocol.json").write_text(
                json.dumps(dict(protocol, control=True))
            )
            (control / "report.json").write_text(json.dumps({"control_passed": True}))
        result["winner"]["gate"] = {"eligible": True}
        return c, out, result, state

    def test_no_partial_or_unmatched_control_activation(self):
        for rid in (
            "0348_aromatic_h_candidate_zero_control",
            "0352_aromatic_h_candidate_mask_system",
        ):
            with tempfile.TemporaryDirectory() as directory:
                c, out, result, state = self.fixture(Path(directory))
                (c.root / "rounds" / rid / "report.json").write_text(
                    json.dumps({"control_passed": False})
                )
                with self.assertRaisesRegex(ValueError, "exact zero and mask"):
                    activate(c, out, result)
                self.assertNotIn("local_verified_inference", state)

    def test_qualified_winner_dispatches_without_changing_original_entry(self):
        from casmi_ml.extended_candidate_activation import resolve_score as dispatcher

        with tempfile.TemporaryDirectory() as directory:
            c, out, result, _state = self.fixture(Path(directory))
            fn = lambda *args: None
            with (
                patch(
                    "casmi_ml.aromatic_h_candidate_activation.load_model",
                    return_value=(fn, {"mask_chemical_channels": False}),
                ),
                patch("casmi_ml.aromatic_h_candidate_activation.write_json"),
            ):
                proof = activate(c, out, result)
                self.assertEqual(
                    proof["extended_candidate_compatibility"]["variant"],
                    "aromatic_original_h",
                )
                self.assertIs(
                    dispatcher({"encoder": "encoder", "critic": "critic"}, proof), fn
                )
                self.assertIs(
                    resolve_score({"encoder": "encoder", "critic": "critic"}, proof), fn
                )
            self.assertIsNone(dispatcher({}, {}))
            self.assertIsNone(activate(None, None, {"winner": None}))


if __name__ == "__main__":
    unittest.main()
