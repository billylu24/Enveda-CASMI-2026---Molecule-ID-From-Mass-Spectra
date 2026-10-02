import tempfile
import unittest
from unittest.mock import patch

from casmi_ml.chemistry import neutral_mass, parse_adduct
from casmi_ml.metfrag_dimer import DimerMetFrag, monomer_product_row


class DimerMetFragTests(unittest.TestCase):
    def row(self, adduct="[2M+Na]+"):
        ion = parse_adduct(adduct)
        mono = parse_adduct(adduct.replace("[2M", "[M", 1))
        marker = 180 + mono.shift
        return {
            "adduct": adduct,
            "precursor_mz": 360 + ion.shift,
            "instrument_type": "Orbitrap",
            "ms2_mzs": [50, marker, marker + 5],
            "ms2_normalized_intensities": [0.8, 0.2, 1],
        }

    def test_observed_marker_enables_only_below_monomer_products(self):
        for adduct in ("[2M+H]+", "[2M-H]-", "[2M+Na]+", "[2M+CH2O2-H]-"):
            row = self.row(adduct)
            transformed = monomer_product_row(row)
            self.assertAlmostEqual(
                neutral_mass(transformed), neutral_mass(row), places=10
            )
            self.assertEqual(transformed["ms2_mzs"], [50])
            self.assertNotIn("normalized_smiles", transformed)

    def test_absent_weak_or_wrong_marker_blocks_java(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = DimerMetFrag(__file__, directory)
            for mode in ("absent", "weak", "wrong"):
                row = self.row()
                if mode == "absent":
                    row["ms2_mzs"] = [50, 60, 70]
                if mode == "weak":
                    row["ms2_normalized_intensities"][1] = 0.01
                if mode == "wrong":
                    row["ms2_mzs"][1] += 0.1
                with patch.object(adapter, "_run") as run:
                    self.assertEqual(adapter.score(row, {"x": "CCO"})["scores"], {})
                    run.assert_not_called()

    def test_multicharged_dimer_is_not_remapped(self):
        row = self.row()
        row["adduct"] = "[2M+2H]2+"
        self.assertIsNone(monomer_product_row(row))
