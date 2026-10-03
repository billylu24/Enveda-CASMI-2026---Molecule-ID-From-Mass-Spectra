import unittest
from unittest.mock import Mock

from casmi_ml.local_proposal_runtime import NAMES, validate_report, verified_factory


class LocalProposalRuntimeTests(unittest.TestCase):
    def test_partial_or_unverified_control_is_rejected(self):
        complete = {
            "runtime_verified": True,
            "runtime_complete_proposal_matches": dict.fromkeys(NAMES, 2000),
        }
        validate_report(complete)
        for report in [
            {},
            {**complete, "runtime_verified": False},
            {
                **complete,
                "runtime_complete_proposal_matches": dict.fromkeys(NAMES, 1999),
            },
        ]:
            with self.assertRaisesRegex(ValueError, "complete2000"):
                validate_report(report)

    def test_remote_mode_and_unactivated_cache_cannot_select_runtime(self):
        controller = Mock()
        controller.local_only.return_value = False
        with self.assertRaisesRegex(ValueError, "local-only"):
            verified_factory(controller)
        controller.local_only.return_value = True
        controller.read.return_value = {}
        with self.assertRaisesRegex(ValueError, "No exact"):
            verified_factory(controller)
