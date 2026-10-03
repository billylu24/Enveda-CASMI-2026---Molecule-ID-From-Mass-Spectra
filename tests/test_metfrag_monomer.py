import tempfile
import unittest
from unittest.mock import patch

from casmi_ml.chemistry import parse_adduct
from casmi_ml.metfrag_monomer import ALIASES, MonomerMetFrag


class MonomerMetFragTests(unittest.TestCase):
    def test_exact_aliases_preserve_charge_and_neutral_mass(self):
        for raw, engine in ALIASES.items():
            a, b = parse_adduct(raw), parse_adduct(engine)
            self.assertEqual((a.molecules, a.charge), (1, b.charge))
            for mass in (100.0, 650.5):
                precursor = mass + a.shift
                self.assertAlmostEqual(b.neutral_mass(precursor), mass, places=10)

    def test_dimer_multicharge_and_dehydration_do_not_execute_java(self):
        with tempfile.TemporaryDirectory() as directory:
            f = MonomerMetFrag(__file__, directory)
            with patch.object(f, "_run") as run:
                for adduct in ("[2M+Na]+", "[M+2H]2+", "[M-H2O+H]+"):
                    self.assertEqual(
                        f.score({"adduct": adduct, "precursor_mz": 200}, {"x": "CCO"})[
                            "scores"
                        ],
                        {},
                    )
                run.assert_not_called()

    def test_alias_reaches_engine_with_original_neutral_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            f = MonomerMetFrag(__file__, directory)
            ion = parse_adduct("[M+CH2O2-H]-")
            row = {
                "adduct": "[M+CH2O2-H]-",
                "precursor_mz": 180.0 + ion.shift,
                "instrument_type": "Orbitrap",
                "ms2_mzs": [45.0],
                "ms2_normalized_intensities": [1.0],
            }
            with patch.object(
                f, "_run", return_value={"status": "complete", "scores": {}}
            ) as run:
                f.score(row, {"x": "CCO"})
                self.assertAlmostEqual(run.call_args.args[3], 180.0)
                self.assertEqual(run.call_args.args[4], "[M+HCOO]-")

    def test_depth_changes_cache_identity_and_invalid_depth_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {
                "adduct": "[M+H]+",
                "precursor_mz": 181.007276,
                "instrument_type": "Orbitrap",
                "ms2_mzs": [45.0],
                "ms2_normalized_intensities": [1.0],
            }
            paths = []
            for depth in (2, 3):
                adapter = MonomerMetFrag(__file__, directory, depth=depth)
                with patch.object(
                    adapter, "_run", return_value={"status": "complete", "scores": {}}
                ) as run:
                    adapter.score(row, {"x": "CCO"})
                    paths.append(run.call_args.args[0])
            self.assertNotEqual(paths[0], paths[1])
            for depth in (0, 4, True, 2.0):
                with self.assertRaises(ValueError):
                    MonomerMetFrag(__file__, directory, depth=depth)
