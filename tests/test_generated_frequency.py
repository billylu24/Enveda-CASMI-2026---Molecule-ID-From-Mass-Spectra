import unittest

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi_ml.generated_frequency import ranked_candidates
from casmi_ml.generation_experiment import validate_generated
from casmi_ml.research_models import SmilesVocabulary


class GeneratedFrequencyTests(unittest.TestCase):
    def test_structure_duplicates_counted_with_best_sequence_preserved(self):
        vocabulary = SmilesVocabulary.fit(["CCO", "OCC", "CCN"])
        sequences = [vocabulary.encode(s) for s in ["CCO", "OCC", "CCN", "CCO"]]
        mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("CCO"))
        args = (
            sequences,
            [-5.0, -2.0, -1.0, -1.0],
            [True, True, True, False],
            vocabulary,
            mass,
            [],
            [],
        )
        legacy, legacy_stats = validate_generated(*args)
        measured, stats = validate_generated(*args, track_frequency=True)
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["sample_count"], 2)
        self.assertEqual(measured[0]["log_probability"], -2)
        self.assertEqual(measured[0]["best_sequence_tokens"], sequences[1].index(2))
        self.assertEqual(
            sum(c["sample_count"] for c in measured), stats["mass_matching"]
        )
        self.assertEqual(legacy_stats, stats)
        self.assertEqual(
            [
                {
                    k: v
                    for k, v in c.items()
                    if k not in ["sample_count", "best_sequence_tokens"]
                }
                for c in measured
            ],
            legacy,
        )

    def test_frequency_ties_preserve_control_and_differing_counts_reorder(self):
        candidates = [{"key": "A", "sample_count": 1}, {"key": "B", "sample_count": 5}]
        self.assertEqual(ranked_candidates(candidates, 0), ["A", "B"])
        self.assertEqual(ranked_candidates(candidates, 1), ["B", "A"])
        self.assertEqual(
            ranked_candidates(
                [{"key": "A", "sample_count": 1}, {"key": "B", "sample_count": 1}], 1
            ),
            ["A", "B"],
        )
        with self.assertRaises(ValueError):
            ranked_candidates([{"key": "A"}], 1)


if __name__ == "__main__":
    unittest.main()
