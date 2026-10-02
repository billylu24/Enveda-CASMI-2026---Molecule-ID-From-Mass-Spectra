import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from casmi_ml.public_catalogs import (
    ColumnSchema, SourceSpec, import_catalog, manifest_path, merge_catalogs,
)


class PublicCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def spec(self, source="pubchemlite", schema=None):
        return SourceSpec(source, "https://example.org/public-download", "CC BY 4.0", schema, "2026-10-01")

    def row(self, smiles="CCO", identifier="001", **changes):
        mol = Chem.MolFromSmiles(smiles)
        row = {
            "Identifier": identifier, "SMILES": smiles,
            "FirstBlock": Chem.MolToInchiKey(mol)[:14] if mol else "",
            "MonoisotopicMass": Descriptors.ExactMolWt(mol) if mol else "",
            "MolecularFormula": rdMolDescriptors.CalcMolFormula(mol) if mol else "",
        }
        row.update(changes)
        return row

    def csv(self, rows, name="input.csv"):
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def sdf(self, records):
        path = self.root / "fixture.sdf"
        writer = Chem.SDWriter(str(path))
        for smiles, properties in records:
            mol = Chem.MolFromSmiles(smiles)
            for name, value in properties.items():
                mol.SetProp(name, str(value))
            writer.write(mol)
        writer.close()
        return path.read_bytes()

    def test_pubchemlite_recomputes_mass_preserves_ids_and_records_hashes(self):
        path = self.csv([self.row(MonoisotopicMass=46.041864813 + 0.001)])
        output = self.root / "out.parquet"
        frame, manifest = import_catalog(path, self.spec(), output)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].source_id, "001")
        self.assertAlmostEqual(frame.iloc[0].mass, Descriptors.ExactMolWt(Chem.MolFromSmiles("CCO")), places=10)
        self.assertEqual(frame.iloc[0].molecular_formula, "C2H6O")
        self.assertEqual(manifest["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest["derived_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(json.loads(manifest_path(output).read_text()), manifest)
        self.assertEqual(json.loads(frame.iloc[0].provenance_json)[0]["license"], "CC BY 4.0")

    def test_invalid_identity_mass_salts_and_empty_structure_are_reported(self):
        rows = [
            self.row("CCO", "valid"), self.row("bogus", "invalid"),
            self.row("CCO", "key", FirstBlock=Chem.MolToInchiKey(Chem.MolFromSmiles("CCN"))[:14]),
            self.row("CCO", "mass", MonoisotopicMass=50),
            self.row("C[NH3+].[Cl-]", "salt"), self.row("", "empty"),
            self.row("*CC", "wildcard"),
        ]
        path = self.csv(rows)
        frame, manifest = import_catalog(path, self.spec())
        self.assertEqual(frame.source_id.tolist(), ["valid"])
        self.assertEqual(manifest["rejections"], {"invalid_structure": 2, "identity_mismatch": 1, "mass_mismatch": 1, "disconnected_structure": 1, "unsupported_atom": 1})
        self.assertEqual(manifest["input_rows"], 7)
        self.assertEqual(manifest["rejected_rows"], 6)

    def test_charged_and_isotopic_structures_are_not_silently_neutralized(self):
        path = self.csv([self.row("C[NH3+]", "charged"), self.row("[13CH3]CO", "isotope")])
        frame, manifest = import_catalog(path, self.spec())
        charged = frame.set_index("source_id").loc["charged"]
        self.assertEqual(charged.formal_charge, 1)
        self.assertIn("+", charged.normalized_smiles)
        self.assertEqual(charged.original_smiles, "C[NH3+]")
        self.assertAlmostEqual(charged.mass, Descriptors.ExactMolWt(Chem.MolFromSmiles("C[NH3+]")))
        self.assertGreater(abs(charged.mass - Descriptors.ExactMolWt(Chem.MolFromSmiles("CN"))), 1)
        self.assertEqual(manifest["charged_structures"], 1)
        self.assertIn("13", frame.set_index("source_id").loc["isotope"].normalized_smiles)

    def test_official_chebi_gz_headers_and_lipidmaps_zip_annotations(self):
        chebi = self.root / "chebi.sdf.gz"
        chebi.write_bytes(gzip.compress(self.sdf([("CCO", {"ChEBI ID": "CHEBI:16236", "ChEBI NAME": "ethanol", "SMILES": "OCC", "INCHIKEY": Chem.MolToInchiKey(Chem.MolFromSmiles("CCO")), "MONOISOTOPIC_MASS": 46.041864813, "FORMULA": "C2H6O"})])))
        frame, _ = import_catalog(chebi, self.spec("chebi"))
        self.assertEqual(frame.iloc[0].original_smiles, "OCC")
        self.assertEqual(frame.iloc[0].source_id, "CHEBI:16236")
        lipid = self.root / "LMSD.sdf.zip"
        raw = self.sdf([("CCO", {"LM_ID": "LMFA-test", "SMILES": "CCO", "INCHI_KEY": Chem.MolToInchiKey(Chem.MolFromSmiles("CCO")), "EXACT_MASS": 46.041864813, "FORMULA": "C2H6O", "MAIN_CLASS": "fixture-class"})])
        with zipfile.ZipFile(lipid, "w") as archive:
            archive.writestr("structures.sdf", raw)
            archive.writestr("README.txt", "do not interpret this as a structure")
        frame, manifest = import_catalog(lipid, self.spec("lipidmaps"))
        self.assertEqual(frame.iloc[0].source_id, "LMFA-test")
        provenance = json.loads(frame.iloc[0].provenance_json)[0]
        self.assertEqual(provenance["annotations"]["MAIN_CLASS"], "fixture-class")
        self.assertEqual(manifest["archive_members"], [{"path": "structures.sdf", "sha256": hashlib.sha256(raw).hexdigest()}])

    def test_pubchem_sdf_and_explicit_tabular_coconut_schema(self):
        sdf = self.root / "pubchem.sdf"
        sdf.write_bytes(self.sdf([("CCO", {"PUBCHEM_COMPOUND_CID": 702, "PUBCHEM_IUPAC_INCHIKEY": Chem.MolToInchiKey(Chem.MolFromSmiles("CCO")), "PUBCHEM_EXACT_MASS": 46.041864813, "PUBCHEM_MOLECULAR_FORMULA": "C2H6O"})]))
        frame, _ = import_catalog(sdf, self.spec("pubchem"))
        self.assertEqual(frame.iloc[0].source_id, "702")
        path = self.csv([{"canonical_smiles": "CCO", "id": "CNP-test", "formula": "C2H6O"}], "coconut.csv")
        with self.assertRaisesRegex(ValueError, "explicit ColumnSchema"):
            import_catalog(path, self.spec("coconut"))
        schema = ColumnSchema(smiles="canonical_smiles", source_id="id", formula="formula")
        frame, _ = import_catalog(path, self.spec("coconut", schema))
        self.assertEqual(frame.iloc[0].source_id, "CNP-test")
        self.assertEqual(frame.iloc[0].origin, "coconut")

    def test_formula_counts_and_charge_mismatches(self):
        schema = ColumnSchema(smiles="SMILES", source_id="Identifier", formula="formula", charge="charge")
        rows = [
            self.row("CCO", "count-order", formula="H6C2O", charge=0),
            self.row("C[O-]", "charged-formula", formula="CH3O-", charge=-1),
            self.row("CCO", "bad-formula", formula="C2H6N", charge=0),
            self.row("C[NH3+]", "bad-charge", formula="CH6N+", charge=0),
        ]
        frame, manifest = import_catalog(self.csv(rows), self.spec("custom", schema))
        self.assertEqual(frame.source_id.tolist(), ["count-order", "charged-formula"])
        self.assertEqual(manifest["rejections"], {"formula_mismatch": 1, "charge_mismatch": 1})

    def test_sdf_smiles_disagreement_is_not_silently_accepted(self):
        path = self.root / "chebi.sdf"
        path.write_bytes(self.sdf([("CCO", {"SMILES": "CCN", "ChEBI ID": "bad"})]))
        frame, manifest = import_catalog(path, self.spec("chebi"))
        self.assertTrue(frame.empty)
        self.assertEqual(manifest["rejections"], {"original_smiles_mismatch": 1})

    def test_sdf_valid_graph_keeps_unparseable_smiles_as_flagged_annotation(self):
        path = self.root / "chebi.sdf"
        path.write_bytes(self.sdf([("CCO", {"SMILES": "invalid-smiles", "ChEBI ID": "fallback", "MONOISOTOPIC_MASS": 46.041864813, "INCHIKEY": Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))})]))
        frame, manifest = import_catalog(path, self.spec("chebi"))
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].normalized_smiles, "CCO")
        self.assertEqual(frame.iloc[0].original_smiles, "CCO")
        provenance = json.loads(frame.iloc[0].provenance_json)[0]
        self.assertEqual(provenance["reported_smiles"], "invalid-smiles")
        self.assertEqual(provenance["source_graph_smiles"], "CCO")
        self.assertEqual(manifest["warnings"], {"source_smiles_parse_failure": 1})

    def test_sdf_stereo_disagreement_preserves_both_source_representations(self):
        path = self.root / "chebi.sdf"
        path.write_bytes(self.sdf([("C[C@H](O)F", {"SMILES": "C[C@@H](O)F", "ChEBI ID": "stereo"})]))
        frame, manifest = import_catalog(path, self.spec("chebi"))
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row.original_smiles, "C[C@@H](O)F")
        self.assertEqual(row.normalized_smiles, Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](O)F")))
        self.assertEqual(manifest["warnings"], {"source_smiles_fullkey_disagreement": 1})
        provenance = json.loads(row.provenance_json)[0]
        self.assertEqual(provenance["source_graph_smiles"], row.normalized_smiles)

    def test_merge_preserves_base_representations_and_unions_sources(self):
        key = Chem.MolToInchiKey(Chem.MolFromSmiles("CCO"))[:14]
        base = pd.DataFrame([{"inchikey14": key, "normalized_smiles": "OCC", "mass": 46.0, "origin": "library"}])
        public, _ = import_catalog(self.csv([self.row("CCO", "a"), self.row("CCO", "b"), self.row("COC", "new")]), self.spec())
        combined = merge_catalogs(base, [public, public])
        self.assertEqual(len(combined), 2)
        row = combined.iloc[0]
        self.assertEqual(row.normalized_smiles, "OCC")
        self.assertEqual(row.mass, 46.0)
        self.assertEqual(row.origin, "library")
        provenance = json.loads(row.provenance_json)
        self.assertEqual([r["source_id"] for r in provenance], [None, "a", "b"])
        self.assertEqual(base.iloc[0].mass, 46.0)

    def test_actual_neutral_source_record_wins_over_charged_first_without_neutralization(self):
        charged, neutral = "CC(=O)[O-]", "CC(=O)O"
        self.assertEqual(Chem.MolToInchiKey(Chem.MolFromSmiles(charged))[:14], Chem.MolToInchiKey(Chem.MolFromSmiles(neutral))[:14])
        frame, manifest = import_catalog(self.csv([self.row(charged, "charged-first"), self.row(neutral, "actual-acid")]), self.spec())
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].source_id, "actual-acid")
        self.assertEqual(frame.iloc[0].formal_charge, 0)
        self.assertEqual(frame.iloc[0].normalized_smiles, neutral)
        self.assertAlmostEqual(frame.iloc[0].mass, Descriptors.ExactMolWt(Chem.MolFromSmiles(neutral)))
        sources = json.loads(frame.iloc[0].provenance_json)
        self.assertEqual([p["formal_charge"] for p in sources], [-1, 0])
        self.assertAlmostEqual(sources[0]["exact_mass"], Descriptors.ExactMolWt(Chem.MolFromSmiles(charged)))
        self.assertEqual(manifest["neutral_representative_replacements"], 1)
        only_charged, _ = import_catalog(self.csv([self.row(charged, "only-charged")]), self.spec())
        self.assertEqual(only_charged.iloc[0].formal_charge, -1)
        merged = merge_catalogs(only_charged, [frame])
        self.assertEqual(merged.iloc[0].source_id, "only-charged")
        self.assertEqual(merged.iloc[0].formal_charge, -1)

    def test_legacy_pubchemlite_charge_is_recovered_without_changing_graph_mass_or_priority(self):
        ethanol, acetate = "OCC", "CC(=O)[O-]"
        legacy = pd.DataFrame([
            {"inchikey14": Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14],
             "normalized_smiles": smiles, "mass": mass, "origin": "pubchemlite"}
            for smiles, mass in [(ethanol, 46.0419), (acetate, 59.0138)]
        ])
        public, _ = import_catalog(self.csv([self.row("CCO", "ethanol-overlap"), self.row("CC(=O)O", "neutral-acid-overlap"), self.row("CCN", "new")]), self.spec())
        merged = merge_catalogs(legacy, [public])
        self.assertEqual(merged.formal_charge.tolist(), [0, -1, 0])
        self.assertEqual(merged.iloc[:2].normalized_smiles.tolist(), [ethanol, acetate])
        self.assertEqual(merged.iloc[:2].mass.tolist(), [46.0419, 59.0138])
        self.assertEqual(merged.iloc[:2].origin.tolist(), ["pubchemlite", "pubchemlite"])
        self.assertEqual(len(merged[merged.formal_charge.eq(0)]), 2)
        self.assertEqual(len(json.loads(merged.iloc[1].provenance_json)), 2)
        self.assertNotIn("formal_charge", legacy.columns)

    def test_legacy_null_charge_is_computed_from_retained_smiles(self):
        public, _ = import_catalog(self.csv([self.row("CCO")]), self.spec())
        legacy = public.copy()
        legacy["formal_charge"] = float("nan")
        merged = merge_catalogs(legacy, [public])
        self.assertEqual(merged.iloc[0].formal_charge, 0)
        self.assertFalse(merged.formal_charge.isna().any())

    def test_missing_legacy_charge_with_invalid_smiles_fails_clearly(self):
        legacy = pd.DataFrame([{"inchikey14": "invalid-source-key", "normalized_smiles": "not-a-smiles", "mass": 46.0, "origin": "pubchemlite"}])
        public, _ = import_catalog(self.csv([self.row()]), self.spec())
        with self.assertRaisesRegex(ValueError, "missing formal_charge.*invalid-source-key.*invalid retained"):
            merge_catalogs(legacy, [public])

    def test_tautomers_keep_original_graphs_and_no_source_license_is_inferred(self):
        path = self.csv([self.row("O=C1C=CC=CN1", "keto"), self.row("Oc1ccccn1", "enol")])
        frame, manifest = import_catalog(path, self.spec())
        represented = {p["original_smiles"] for raw in frame.provenance_json for p in json.loads(raw)}
        self.assertEqual(represented, {"O=C1C=CC=CN1", "Oc1ccccn1"})
        self.assertEqual(manifest["valid_rows"], 2)
        for url, license in [("", "CC0"), ("https://example.org/", "")]:
            with self.assertRaises(ValueError):
                SourceSpec("coconut", url, license)
        bad_spec = SourceSpec("pubchemlite", "https://example.org/", "CC0", expected_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            import_catalog(path, bad_spec)

    def test_cli_import_and_merge_create_readable_lineage(self):
        path = self.csv([self.row()])
        output, combined = self.root / "out.parquet", self.root / "combined.parquet"
        result = subprocess.run([sys.executable, "-m", "casmi_ml.public_catalogs", "import", str(path), "--source", "pubchemlite", "--source-url", "https://example.org/public", "--license", "CC BY 4.0", "--output", str(output)], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["structures"], 1)
        subprocess.run([sys.executable, "-m", "casmi_ml.public_catalogs", "merge", str(output), str(output), "--output", str(combined)], check=True, capture_output=True, text=True)
        manifest = json.loads(manifest_path(combined).read_text())
        self.assertEqual(manifest["structures"], 1)
        self.assertIsNotNone(manifest["inputs"][0]["manifest_sha256"])
        self.assertEqual(pd.read_parquet(combined).iloc[0].source_id, "001")

    def test_external_only_union_selects_actual_neutral_from_second_source(self):
        schema = ColumnSchema(smiles="SMILES", source_id="Identifier", inchikey="FirstBlock", exact_mass="MonoisotopicMass", formula="MolecularFormula")
        outputs = []
        for source, smiles in [("chebi", "CC(=O)[O-]"), ("lipidmaps", "CC(=O)O")]:
            path = self.csv([self.row(smiles, source)], f"{source}.csv")
            output = self.root / f"{source}.parquet"
            import_catalog(path, self.spec(source, schema), output)
            outputs.append(output)
        result = self.root / "public-union.parquet"
        subprocess.run([sys.executable, "-m", "casmi_ml.public_catalogs", "merge", *(str(p) for p in outputs), "--prefer-source-neutral", "--output", str(result)], check=True, capture_output=True, text=True)
        frame = pd.read_parquet(result)
        self.assertEqual(frame.iloc[0].formal_charge, 0)
        self.assertEqual(frame.iloc[0].source_id, "lipidmaps")
        self.assertEqual([p["source"] for p in json.loads(frame.iloc[0].provenance_json)], ["chebi", "lipidmaps"])
        self.assertEqual(json.loads(manifest_path(result).read_text())["neutral_representative_replacements"], 1)


if __name__ == "__main__":
    unittest.main()
