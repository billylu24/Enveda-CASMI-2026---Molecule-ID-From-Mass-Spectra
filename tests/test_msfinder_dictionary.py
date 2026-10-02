import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from casmi_ml.msfinder_dictionary import (
    ELECTRON_MASS, HEADERS, dictionary_manifest, load_msfinder_dictionary,
    main, match_msfinder_dictionary,
)


class MsfinderDictionaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "dictionary.tsv"

    def write_rows(self, rows, header=HEADERS):
        self.path.write_text("\t".join(header) + "\n" +
                             "\n".join("\t".join(map(str, row)) for row in rows) + "\n")
        return self.path

    def load(self, kind="diagnostic_ion", convention="msfinder_atomic"):
        return load_msfinder_dictionary(self.path, kind, mass_convention=convention)

    def test_upstream_headers_modes_keys_and_explicit_electron_correction(self):
        # Two real upstream records; no raw database is bundled in the repo.
        self.write_rows([
            (41.03912516, "C3H5", "Negative", 0.19, "ATUOYWHBWRKTHZ"),
            (41.03912516, "C3H5", "Positive", 0.87, "ATUOYWHBWRKTHZ;QQONPFPTGQHPMA"),
        ])
        negative, positive = self.load()
        self.assertEqual(negative["mode"], "negative")
        self.assertEqual(positive["kind"], "diagnostic_ion")
        self.assertAlmostEqual(negative["mass"], 41.03912516 + ELECTRON_MASS)
        self.assertAlmostEqual(positive["mass"], 41.03912516 - ELECTRON_MASS)
        self.assertEqual(positive["stored_mass"], 41.03912516)
        self.assertEqual(positive["fragment_keys"], ["ATUOYWHBWRKTHZ", "QQONPFPTGQHPMA"])
        self.assertEqual(positive["frequency"], 0.87)
        self.assertNotIn("smarts", positive)

    def test_requires_mass_convention_and_rejects_wrong_convention(self):
        self.write_rows([(30.03437413, "CH4N", "Positive", 0.14, "BAVYZALUXZFZLV")])
        with self.assertRaisesRegex(ValueError, "explicit mass_convention"):
            load_msfinder_dictionary(self.path, "product_ion")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.load(convention="already_mz")
        corrected = 30.03437413 - ELECTRON_MASS
        self.write_rows([(corrected, "CH4N", "Positive", 0.14, "BAVYZALUXZFZLV")])
        rule = self.load(convention="already_mz")[0]
        self.assertEqual(rule["mass"], corrected)
        self.assertEqual(rule["mass_correction_da"], 0)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.load()

    def test_neutral_loss_mass_is_unchanged_in_both_modes(self):
        self.write_rows([(18.01056468, "H2O", mode, 10, "XLYOFNOQVPJJNP")
                         for mode in ("Positive", "Negative")])
        for convention in ("msfinder_atomic", "neutral_mass"):
            for rule in self.load("neutral_loss", convention):
                self.assertEqual(rule["mass"], 18.01056468)
                self.assertEqual(rule["mass_correction_da"], 0)
        with self.assertRaisesRegex(ValueError, "not ion m/z"):
            self.load("neutral_loss", "already_mz")

    def test_malformed_rows_do_not_produce_partial_import(self):
        valid = (18.01056468, "H2O", "Positive", 10, "XLYOFNOQVPJJNP")
        bad_rows = [valid[:-1], valid + ("extra",),
                    ("nan", *valid[1:]), ("inf", *valid[1:]),
                    (18.01056468, "H0O", *valid[2:]),
                    (18.01056468, "H2O", "unknown", *valid[3:]),
                    (*valid[:3], "", valid[4]),
                    (*valid[:3], -1, valid[4]),
                    (*valid[:4], "invalid-key"),
                    (19.0, *valid[1:])]
        for row in bad_rows:
            with self.subTest(row=row):
                self.write_rows([valid, row])
                with self.assertRaisesRegex(ValueError, ":3:"):
                    self.load("neutral_loss")
        self.write_rows([valid], header=("Mass", *HEADERS[1:]))
        with self.assertRaisesRegex(ValueError, ":1:"):
            self.load("neutral_loss")

    def test_mode_separation_and_peak_threshold(self):
        self.write_rows([(41.03912516, "C3H5", mode, 1, "ATUOYWHBWRKTHZ")
                         for mode in ("Positive", "Negative")])
        rules = self.load()
        positive = rules[0]["mass"]
        # Tight tolerance also distinguishes the 1.097 mDa polarity separation.
        annotations = match_msfinder_dictionary([positive, positive], [1, 0.001], rules,
                                               mode="Positive", ppm=1, da_floor=0.0001)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["mode"], "positive")
        self.assertEqual(annotations[0]["peak_index"], 0)
        self.assertEqual(annotations[0]["evidence_type"], "mass_match_only")
        self.assertEqual(match_msfinder_dictionary([positive], [1], rules, mode="Negative",
                                                  ppm=1, da_floor=0.0001), [])

    def test_mass_tolerance_and_neutral_loss_error_propagation(self):
        loss = 18.01056468
        self.write_rows([(loss, "H2O", "Positive", 2, "XLYOFNOQVPJJNP")])
        rules = self.load("neutral_loss")
        precursor = 500.0
        annotations = match_msfinder_dictionary([precursor - loss + 0.008], [2], rules,
                                               mode="positive", precursor_mz=precursor,
                                               ppm=10, da_floor=0.002)
        self.assertEqual(len(annotations), 1)
        self.assertAlmostEqual(annotations[0]["error_da"], -0.008)
        self.assertGreater(annotations[0]["tolerance_da"], 0.009)
        self.assertEqual(match_msfinder_dictionary([precursor - loss + 0.011], [2], rules,
                                                  mode="positive", precursor_mz=precursor,
                                                  ppm=10, da_floor=0.002), [])
        self.assertEqual(match_msfinder_dictionary([precursor - loss], [2], rules,
                                                  mode="positive"), [])
        with self.assertRaisesRegex(ValueError, "precursor_charge=1"):
            match_msfinder_dictionary([100], [1], rules, mode="positive",
                                     precursor_mz=precursor, precursor_charge=2)

    def test_no_annotations_for_empty_or_zero_spectrum_and_invalid_inputs(self):
        self.write_rows([(18.01056468, "H2O", "Positive", 10, "XLYOFNOQVPJJNP")])
        rules = self.load("neutral_loss")
        self.assertEqual(match_msfinder_dictionary([], [], rules, mode="positive"), [])
        self.assertEqual(match_msfinder_dictionary([100], [0], rules, mode="positive"), [])
        with self.assertRaisesRegex(ValueError, "same length"):
            match_msfinder_dictionary([100], [], rules, mode="positive")
        with self.assertRaisesRegex(ValueError, "invalid peak"):
            match_msfinder_dictionary([float("nan")], [1], rules, mode="positive")

    def test_cli_requires_provenance_and_records_license_status(self):
        self.write_rows([(18.01056468, "H2O", "Positive", 10, "XLYOFNOQVPJJNP")])
        rules = self.load("neutral_loss")
        with self.assertRaisesRegex(ValueError, "source_uri"):
            dictionary_manifest(self.path, rules, source_uri="", license_status="unverified")
        with self.assertRaisesRegex(ValueError, "license_uri"):
            dictionary_manifest(self.path, rules, source_uri="https://example.org/db",
                                license_status="verified")
        output = self.root / "normalized.json"
        spectra = self.root / "spectra.json"
        spectra.write_text(json.dumps([{"spectrum_id": "test", "mzs": [100 - 18.01056468],
                                        "intensities": [1], "mode": "positive", "precursor_mz": 100}]))
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--input", str(self.path), "--kind", "neutral_loss",
                  "--mass-convention", "msfinder_atomic", "--source-uri", "https://example.org/db",
                  "--license-status", "unverified", "--output", str(output), "--spectra", str(spectra)])
        result = json.loads(output.read_text())
        self.assertEqual(result["manifest"]["license_status"], "unverified")
        self.assertEqual(result["manifest"]["source_uri"], "https://example.org/db")
        self.assertEqual(len(result["manifest"]["source_sha256"]), 64)
        self.assertEqual(len(result["annotations"][0]["matches"]), 1)
        self.assertNotIn("functional_group", result["annotations"][0]["matches"][0])


if __name__ == "__main__":
    unittest.main()
