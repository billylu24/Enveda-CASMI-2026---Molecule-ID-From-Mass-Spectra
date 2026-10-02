import json
from pathlib import Path
import sys

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from baseline import PROTON, vectorize
from hybrid import predict as historical_predict
from casmi_ml.chemical_priors import load_rules
from casmi_ml.hybrid_chemistry import (
    DEFAULT_WEIGHT, HybridChemistry, behavior_check, diagnostic_queries,
    predict, prepare_analog_pool,
)


DICTIONARY = Path(__file__).resolve().parents[1] / "configs/chemical_priors.json"
ORIGINAL = "COCS(=O)(=O)O"
SULFATE = "CCOS(=O)(=O)O"


def key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]


def fixture():
    mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(ORIGINAL))
    assert mass == Descriptors.ExactMolWt(Chem.MolFromSmiles(SULFATE))
    records = [(mass, key(ORIGINAL), ORIGINAL, vectorize([80., 100.], [1., .8]))]
    coconut = pd.DataFrame([{"inchikey": key(ORIGINAL), "canonical_smiles": ORIGINAL, "exact_mass": mass}])
    catalog = pd.DataFrame([{"inchikey14": key(SULFATE), "normalized_smiles": SULFATE,
                             "mass": mass, "formal_charge": 0}])
    precursor = mass + PROTON
    test = pd.DataFrame([{"molecule_id": name, "precursor_mz": precursor, "adduct": "[M+H]+",
                          "ionization_mode": "positive", "ms2_mzs": mz,
                          "ms2_normalized_intensities": intensity}
                         for name, mz, intensity in [
                             ("protected", [80., 100.], [1., .8]),
                             ("low", [precursor - 79.956815], [1.]),
                         ]])
    return mass, records, coconut, catalog, test


def test_guard_and_both_disabled_reproduce_historical_exactly():
    _, records, coconut, catalog, test = fixture()
    expected = historical_predict(test, records, coconut)
    engine = HybridChemistry(records, coconut, catalog, load_rules(DICTIONARY), chemistry_weight=1.)
    result, audits = engine.predict(test, catalog_enabled=False, rules_enabled=False)
    assert result.set_index("molecule_id").smiles.to_dict() == expected
    expanded, audits = engine.predict(test)
    rows = expanded.set_index("molecule_id")
    assert rows.loc["protected", "smiles"] == expected["protected"]
    protected = next(a for a in audits if a["molecule_id"] == "protected")
    assert protected["protected"] and protected["chemical_rules_matched"] == 0
    assert protected["new_output_candidates"] == 0
    assert not protected["changed_historical_ranking"]


def test_new_isomer_enters_only_enabled_pool_and_rules_can_promote_motif():
    _, records, coconut, catalog, test = fixture()
    engine = HybridChemistry(records, coconut, catalog, load_rules(DICTIONARY), chemistry_weight=1.)
    variants = engine.variants(test[test.molecule_id.eq("low")], {
        "A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)})
    assert {k for k, _ in variants["A"][0]} == {key(ORIGINAL)}
    assert {k for k, _ in variants["B"][0]} == {key(ORIGINAL), key(SULFATE)}
    assert {k for k, _ in variants["C"][0]} == {key(ORIGINAL)}
    assert variants["D"][0][0] == (key(SULFATE), SULFATE)
    audit = variants["D"][1]
    assert audit["new_mass_candidates"] == audit["new_output_candidates"] == 1
    assert audit["chemical_rules_matched"] == 1 and audit["chemistry_changed_ranking"]


def test_public_catalog_cannot_replace_old_graph_mass_and_excludes_charge():
    mass, _, coconut, catalog, _ = fixture()
    additions = pd.concat([catalog, pd.DataFrame([
        {"inchikey14": key(ORIGINAL), "normalized_smiles": SULFATE, "mass": mass + 1, "formal_charge": 0},
        {"inchikey14": key("C[N+](C)(C)C"), "normalized_smiles": "C[N+](C)(C)C", "mass": 74., "formal_charge": 1},
    ])], ignore_index=True)
    combined, new_keys, report = prepare_analog_pool(coconut, additions)
    assert combined.iloc[0].to_dict() == coconut.iloc[0].to_dict()
    assert new_keys == {key(SULFATE)} and report["charged_or_unknown_excluded"] == 1
    assert len(combined) == 2
    # Legacy missing charges are inferred from the original graph, never dropped en masse.
    legacy = additions.drop(columns="formal_charge")
    inferred, _, report = prepare_analog_pool(coconut, legacy)
    assert inferred.inchikey.tolist() == combined.inchikey.tolist()
    assert report["charged_or_unknown_excluded"] == 1


def test_public_catalog_projects_fields_before_aliases_are_renamed():
    mass, _, coconut, catalog, _ = fixture()
    # Full public catalogs preserve fields whose names overlap the analog schema.
    catalog = catalog.assign(inchikey="source-full-key", canonical_smiles="C",
                             exact_mass=1., provenance_json='{"source": "example"}')
    combined, new_keys, report = prepare_analog_pool(coconut, catalog)
    assert combined.columns.tolist() == ["inchikey", "canonical_smiles", "exact_mass"]
    assert combined.iloc[0].to_dict() == coconut.iloc[0].to_dict()
    assert combined.iloc[1].to_dict() == {
        "inchikey": key(SULFATE), "canonical_smiles": SULFATE, "exact_mass": mass}
    assert new_keys == {key(SULFATE)} and report["appended_structures"] == 1


def write_train(path, mass, rows):
    result = []
    for smiles, peaks, source in rows:
        mol = Chem.MolFromSmiles(smiles)
        result.append({"inchikey14": key(smiles), "normalized_smiles": smiles,
                       "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
                       "precursor_error_ppm": 0., "ingest_lib": source,
                       "precursor_mz": mass + PROTON, "adduct": "[M+H]+", "ionization_mode": "positive",
                       "ms2_mzs": peaks, "ms2_normalized_intensities": [1.] * len(peaks)})
    pd.DataFrame(result).to_parquet(path, index=False)


def test_full_offline_predict_writes_reports_and_defaults_are_predetermined(tmp_path):
    mass, _, coconut, catalog, test = fixture()
    write_train(tmp_path / "train.parquet", mass, [(ORIGINAL, [80., 100.], "library")])
    test.to_parquet(tmp_path / "test.parquet")
    coconut.to_parquet(tmp_path / "coconut.parquet")
    catalog.to_parquet(tmp_path / "catalog.parquet")
    output = tmp_path / "submission.csv"
    submission, report = predict(tmp_path, tmp_path / "coconut.parquet", tmp_path / "catalog.parquet",
                                 DICTIONARY, output)
    assert report["chemistry_weight"] == DEFAULT_WEIGHT == .1
    assert report["protected_molecules"] == report["low_confidence_molecules"] == 1
    assert report["new_output_candidates"] == 1 and report["rule_matched_molecules"] == 1
    assert len(submission) == 2
    assert json.loads(Path(str(output) + ".report.json").read_text())["status"].endswith("unvalidated")
    assert Path(str(output) + ".routing.csv").exists()
    assert Path(str(output) + ".evidence.json").exists()
    # The controls need neither an external catalog nor a dictionary file.
    _, report = predict(tmp_path, tmp_path / "coconut.parquet", tmp_path / "missing.parquet",
                        tmp_path / "missing.json", tmp_path / "control.csv",
                        catalog_enabled=False, rules_enabled=False)
    assert report["catalog_sha256"] is None and report["dictionary_sha256"] is None


def test_real_source_behavior_mask_excludes_keys_across_sources_and_four_controls(tmp_path):
    mass, _, coconut, catalog, _ = fixture()
    signal = mass + PROTON - 79.956815
    # Sulfate query occurs in both source and another library: both references must be removed.
    write_train(tmp_path / "train.parquet", mass, [
        (SULFATE, [signal], "enveda-np-examples"),
        (SULFATE, [signal, 90.], "other"),
        (ORIGINAL, [80., 100.], "other"),
    ])
    coconut.to_parquet(tmp_path / "coconut.parquet")
    catalog.to_parquet(tmp_path / "catalog.parquet")
    queries, selected = diagnostic_queries(tmp_path / "train.parquet", count=32)
    assert selected == [key(SULFATE)] and len(queries) == 1
    report = behavior_check(tmp_path / "train.parquet", tmp_path / "coconut.parquet",
                            tmp_path / "catalog.parquet", DICTIONARY, tmp_path / "behavior.json",
                            chemistry_weight=1.)
    assert report["remaining_reference_spectra"] == 1
    assert report["status"] == "behavior_only_not_independent_validation"
    assert len(report["groups"]) == 4
    assert report["groups"]["A_original_no_rules"]["mrr25"] == 0
    assert report["groups"]["D_expanded_rules"]["mrr25"] == 1


def test_diagnostic_scan_reads_only_query_fields_and_preserves_selection(tmp_path, monkeypatch):
    mass, _, _, _, _ = fixture()
    path = tmp_path / "train.parquet"
    write_train(path, mass, [(SULFATE, [70.], "enveda-np-examples"),
                            (SULFATE, [90.], "enveda-np-examples"),
                            (ORIGINAL, [80.], "other")])
    full = pd.read_parquet(path).assign(spectrum_id=["first", "repeat", "other"],
                                      spectrum_embedding=[[0.] * 16] * 3)
    full.to_parquet(path)
    from casmi_ml import hybrid_chemistry
    original_parquet_file = hybrid_chemistry.pq.ParquetFile
    scans = []

    class TrackedParquetFile:
        def __init__(self, source):
            self.parquet = original_parquet_file(source)
            self.schema_arrow = self.parquet.schema_arrow

        def iter_batches(self, **kwargs):
            scans.append(kwargs.get("columns"))
            return self.parquet.iter_batches(**kwargs)

    monkeypatch.setattr(hybrid_chemistry.pq, "ParquetFile", TrackedParquetFile)
    queries, selected = diagnostic_queries(path)
    assert selected == [key(SULFATE)]
    assert queries.spectrum_id.tolist() == ["first"]
    assert queries.iloc[0].ms2_mzs.tolist() == [70.]
    assert all(columns is not None and "spectrum_embedding" not in columns for columns in scans)
    assert "spectrum_embedding" not in queries.columns


def test_rejects_invalid_weight_and_empty_reference():
    _, records, coconut, catalog, _ = fixture()
    with pytest.raises(ValueError, match="chemistry_weight"):
        HybridChemistry(records, coconut, catalog, chemistry_weight=float("nan"))
    with pytest.raises(ValueError, match="reference"):
        HybridChemistry([], coconut)


def test_module_import_does_not_import_torch():
    # A subprocess ensures torch imported by unrelated existing tests cannot hide dependencies.
    import subprocess
    result = subprocess.run([sys.executable, "-c",
                             "import casmi_ml.hybrid_chemistry,sys; assert 'torch' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
