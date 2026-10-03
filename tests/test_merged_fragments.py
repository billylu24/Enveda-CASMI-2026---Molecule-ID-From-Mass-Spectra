import unittest

from casmi_ml.merged_fragments import merge_records


class MergedFragmentTests(unittest.TestCase):
    def row(self, mz, intensity, adduct="[M+H]+", precursor=181.007276):
        return {
            "adduct": adduct,
            "precursor_mz": precursor,
            "instrument_type": "Orbitrap",
            "ms2_mzs": mz,
            "ms2_normalized_intensities": intensity,
            "normalized_smiles": "must_not_enter_merged_query",
            "molecular_formula": "C6H12O6",
        }

    def test_union_retains_observed_masses_max_intensity_and_removes_labels(self):
        rows = [
            self.row([80, 100, 181], [0.5, 1, 0.1]),
            self.row([100.001, 120], [0.6, 1]),
        ]
        merged = merge_records(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["ms2_mzs"], [80, 100, 120])
        self.assertEqual(merged[0]["ms2_normalized_intensities"], [0.5, 1, 1])
        self.assertNotIn("normalized_smiles", merged[0])
        self.assertNotIn("molecular_formula", merged[0])

    def test_different_ions_or_inconsistent_neutral_masses_never_merge(self):
        rows = [
            self.row([80], [1]),
            self.row([90], [1], "[M-H]-", 178.9927),
            self.row([70], [1], precursor=190),
        ]
        self.assertEqual(len(merge_records(rows)), 3)
