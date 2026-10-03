import unittest

import pandas as pd

from casmi_ml.cached_submission_validation import (
    checked_structure_key,
    validate_submission,
)
from casmi_ml.inference import validate_submission as original


class CachedValidationTests(unittest.TestCase):
    def test_every_failure_path_matches_original_and_warm_valid_inputs(self):
        test = pd.DataFrame({"molecule_id": ["a", "b"]})
        cases = [
            [
                {"molecule_id": "a", "smiles": "CCC"},
                {"molecule_id": "b", "smiles": "CCO"},
            ],
            [
                {"molecule_id": "a", "smiles": "CCC"},
                {"molecule_id": "a", "smiles": "CCO"},
            ],
            [{"molecule_id": "a", "smiles": "CCC"}],
            [{"molecule_id": "a", "smiles": ""}, {"molecule_id": "b", "smiles": "CCO"}],
            [
                {"molecule_id": "a", "smiles": None},
                {"molecule_id": "b", "smiles": "CCO"},
            ],
            [
                {"molecule_id": "a", "smiles": "invalid"},
                {"molecule_id": "b", "smiles": "CCO"},
            ],
            [
                {"molecule_id": "a", "smiles": "CCC;CCC"},
                {"molecule_id": "b", "smiles": "CCO"},
            ],
            [
                {"molecule_id": "a", "smiles": "C[C@H](O)C(=O)O;C[C@@H](O)C(=O)O"},
                {"molecule_id": "b", "smiles": "CCO"},
            ],
            [
                {"molecule_id": "a", "smiles": ";".join(["C"] * 26)},
                {"molecule_id": "b", "smiles": "CCO"},
            ],
        ]
        for rows in cases:
            outcomes = []
            for fn in [original, validate_submission, validate_submission]:
                try:
                    outcomes.append(fn(test, pd.DataFrame(rows)))
                except ValueError as exc:
                    outcomes.append(str(exc))
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(outcomes[0], outcomes[2])

    def test_pure_cache_is_bounded_and_preserves_actual_connectivity(self):
        checked_structure_key.cache_clear()
        self.assertEqual(checked_structure_key("CCO"), checked_structure_key("OCC"))
        self.assertIsNone(checked_structure_key(""))
        checked_structure_key("CCO")
        self.assertGreater(checked_structure_key.cache_info().hits, 0)
        self.assertEqual(checked_structure_key.cache_info().maxsize, 100000)
