"""CPU-only experimental public-catalog and chemistry extension of hybrid.py.

The historical library/COCONUT result is preserved at confidence >= 0.5. Below
that threshold, additional neutral public structures share the original analog
scoring, followed by optional positive-only chemical reranking of the blended
25 candidates. The default chemistry weight (0.1) is predetermined, not fitted.
No neural checkpoint, torch dependency, or performance improvement is claimed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from baseline import ADDUCT_MASS, load_candidates, make_matrix
from hybrid import blend, coconut_rank, library_rank
from casmi_ml.chemical_priors import (
    candidate_scores, dictionary_sha256, extract_evidence, load_rules, rerank,
)


COCONUT_COLUMNS = ["inchikey", "canonical_smiles", "exact_mass"]
DEFAULT_WEIGHT = 0.1
GUARD_THRESHOLD = 0.5


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _locate(path, filename):
    path = Path(path)
    if path.exists():
        return path
    matches = list(Path("/kaggle/input").rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one mounted {filename}, found {len(matches)}")
    return matches[0]


def prepare_analog_pool(coconut, catalog):
    """Append neutral new keys, preserving every original COCONUT row/value."""
    old = coconut[COCONUT_COLUMNS].copy()
    if catalog is None:
        return old, set(), {"input_structures": 0, "charged_or_unknown_excluded": 0,
                            "invalid_excluded": 0, "appended_structures": 0}
    required = {"inchikey14", "normalized_smiles", "mass"}
    if not required <= set(catalog.columns):
        raise ValueError(f"Public catalog requires columns {sorted(required)}")
    extra = catalog.copy()
    # Legacy public catalogs may lack this field. Infer only from their preserved
    # graph; never neutralize it or treat unknown charge as neutral.
    if "formal_charge" not in extra.columns:
        extra["formal_charge"] = np.nan
    missing = extra.formal_charge.isna()
    if missing.any():
        def charge(smiles):
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            return Chem.GetFormalCharge(mol) if mol is not None and mol.GetNumAtoms() else np.nan
        extra.loc[missing, "formal_charge"] = extra.loc[missing, "normalized_smiles"].map(charge)
    neutral = extra.formal_charge.eq(0)
    excluded_charge = int((~neutral).sum())
    extra = extra.loc[neutral].copy()
    extra["mass"] = pd.to_numeric(extra.mass, errors="coerce")
    valid = (extra.inchikey14.map(lambda key: isinstance(key, str) and len(key) == 14)
             & extra.normalized_smiles.map(lambda smi: isinstance(smi, str) and bool(smi))
             & np.isfinite(extra.mass) & extra.mass.gt(0))
    invalid = int((~valid).sum())
    extra = extra.loc[valid].drop_duplicates("inchikey14", keep="first")
    old_keys = set(old.inchikey.str[:14])
    extra = extra.loc[~extra.inchikey14.isin(old_keys)]
    new_keys = set(extra.inchikey14)
    extra = extra.rename(columns={"inchikey14": "inchikey", "normalized_smiles": "canonical_smiles", "mass": "exact_mass"})
    unified = pd.concat([old, extra[COCONUT_COLUMNS]], ignore_index=True)
    return unified, new_keys, {
        "input_structures": len(catalog), "charged_or_unknown_excluded": excluded_charge,
        "invalid_excluded": invalid, "appended_structures": len(extra),
    }


def validate_submission(test, submission):
    """Validate IDs, count, SMILES, and raw InChIKey14 uniqueness without torch."""
    if (set(submission.molecule_id) != set(test.molecule_id)
            or submission.molecule_id.duplicated().any()):
        raise ValueError("Submission molecule IDs do not match the test set")
    for row in submission.itertuples():
        smiles = row.smiles.split(";") if isinstance(row.smiles, str) else []
        if not 1 <= len(smiles) <= 25:
            raise ValueError(f"Invalid candidate count: {row.molecule_id}")
        keys = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or not mol.GetNumAtoms():
                raise ValueError(f"Invalid SMILES: {row.molecule_id}")
            keys.append(Chem.MolToInchiKey(mol)[:14])
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate structures: {row.molecule_id}")


class HybridChemistry:
    """Shared reference and analog indexes for deployment and four controls."""

    def __init__(self, records, coconut, catalog=None, rules=(), chemistry_weight=DEFAULT_WEIGHT):
        if not math.isfinite(chemistry_weight) or not 0 <= chemistry_weight <= 1:
            raise ValueError("chemistry_weight must be finite and between zero and one")
        if not records:
            raise ValueError("No retained reference spectra; historical analog scoring needs references")
        self.records = records
        self.masses = np.asarray([record[0] for record in records])
        self.order = np.argsort(self.masses)
        self.matrix = make_matrix([record[3] for record in records])
        self.coconut = coconut[COCONUT_COLUMNS].copy()
        self.unified, self.new_keys, self.catalog_report = prepare_analog_pool(coconut, catalog)
        self.rules = tuple(rules)
        self.weight = chemistry_weight
        self.cache = {}
        self.old_order = np.argsort(self.coconut.exact_mass.to_numpy())
        self.old_mass = self.coconut.exact_mass.to_numpy()[self.old_order]
        self.new_order = np.argsort(self.unified.exact_mass.to_numpy())
        self.new_mass = self.unified.exact_mass.to_numpy()[self.new_order]

    def variants(self, group, requests):
        """Evaluate named (catalog_enabled, rules_enabled) controls on one query."""
        center, library = library_rank(group, self.records, self.matrix, self.masses, self.order)
        old_analog = coconut_rank(center, library, self.coconut, self.old_mass, self.old_order, self.cache)
        historical = blend(library, old_analog)
        confidence = float(library[0][2]) if library else 0.0
        protected = confidence >= GUARD_THRESHOLD
        use_new = not protected and any(catalog for catalog, _ in requests.values())
        new_analog = (coconut_rank(center, library, self.unified, self.new_mass, self.new_order, self.cache)
                      if use_new else old_analog)
        new_candidates = {key for key, _, _ in new_analog} - {key for key, _, _ in old_analog} - {key for key, _, _ in library}
        use_rules = not protected and any(chemistry for _, chemistry in requests.values())
        evidence = extract_evidence(group, self.rules) if use_rules else []
        results = {}
        for name, (catalog_enabled, rules_enabled) in requests.items():
            pairs = historical if protected else blend(library, new_analog if catalog_enabled else old_analog)
            before = [key for key, _ in pairs]
            structures = dict(pairs)
            active_evidence = evidence if rules_enabled and not protected else []
            scores, support = candidate_scores(structures, active_evidence, self.rules) if active_evidence else ({}, {})
            ranking = rerank(before, scores, self.weight) if rules_enabled and not protected else before
            output = [(key, structures[key]) for key in ranking]
            audit = {
                "confidence": confidence, "protected": protected,
                "catalog_enabled": catalog_enabled, "rules_enabled": rules_enabled,
                "new_mass_candidates": len(new_candidates) if catalog_enabled and not protected else 0,
                "new_output_candidates": sum(key in new_candidates for key in ranking) if catalog_enabled and not protected else 0,
                "chemical_rules_matched": len(active_evidence),
                "catalog_changed_ranking": before != [key for key, _ in historical],
                "chemistry_changed_ranking": ranking != before,
                "changed_historical_ranking": ranking != [key for key, _ in historical],
                "evidence": active_evidence,
                "top25": [{"key": key, "score": scores.get(key, 0.0), "supported_rules": support.get(key, [])}
                          for key in ranking],
            }
            results[name] = (output, audit)
        return results

    def predict(self, test, *, catalog_enabled=True, rules_enabled=True):
        rows, audits = [], []
        for molecule_id, group in test.groupby("molecule_id", sort=False):
            pairs, audit = self.variants(group, {"selected": (catalog_enabled, rules_enabled)})["selected"]
            rows.append({"molecule_id": molecule_id, "smiles": ";".join(smi for _, smi in pairs)})
            audits.append({"molecule_id": molecule_id, **audit})
        submission = pd.DataFrame(rows)
        validate_submission(test, submission)
        return submission, audits


def _summary(audits):
    return {
        "molecules": len(audits),
        "protected_molecules": sum(row["protected"] for row in audits),
        "low_confidence_molecules": sum(not row["protected"] for row in audits),
        "new_mass_candidates": sum(row["new_mass_candidates"] for row in audits),
        "new_output_candidates": sum(row["new_output_candidates"] for row in audits),
        "rule_matched_molecules": sum(bool(row["chemical_rules_matched"]) for row in audits),
        "rule_matches": sum(row["chemical_rules_matched"] for row in audits),
        "catalog_changed_molecules": sum(row["catalog_changed_ranking"] for row in audits),
        "chemistry_changed_molecules": sum(row["chemistry_changed_ranking"] for row in audits),
        "changed_historical_molecules": sum(row["changed_historical_ranking"] for row in audits),
    }


def predict(data_dir, coconut_path, catalog_path, dictionary_path, output, *,
            chemistry_weight=DEFAULT_WEIGHT, catalog_enabled=True, rules_enabled=True):
    started = time.monotonic()
    train_path = _locate(Path(data_dir) / "train.parquet", "train.parquet")
    test_path = _locate(Path(data_dir) / "test.parquet", "test.parquet")
    coconut_path = _locate(coconut_path, "coconut_structures.parquet")
    test = pd.read_parquet(test_path)
    coconut = pd.read_parquet(coconut_path, columns=COCONUT_COLUMNS)
    catalog = pd.read_parquet(catalog_path) if catalog_enabled else None
    rules = load_rules(dictionary_path) if rules_enabled else ()
    neutral = test.precursor_mz.to_numpy() - test.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral) & (neutral > 0)]
    if not len(neutral):
        raise ValueError("No supported query precursor masses")
    records, _ = load_candidates(train_path, neutral)
    engine = HybridChemistry(records, coconut, catalog, rules, chemistry_weight)
    submission, audits = engine.predict(test, catalog_enabled=catalog_enabled, rules_enabled=rules_enabled)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    pd.DataFrame([{key: value for key, value in audit.items() if key not in {"evidence", "top25"}}
                  for audit in audits]).to_csv(str(output) + ".routing.csv", index=False)
    _write_json(str(output) + ".evidence.json", audits)
    report = {
        "status": "experimental_historical_hybrid_extension_unvalidated",
        "base": "historical hybrid.py; no recovered neural checkpoint",
        "guard_threshold": GUARD_THRESHOLD, "chemistry_weight": chemistry_weight,
        "weight_selection": "predetermined; not selected on behavior checks",
        "catalog_enabled": catalog_enabled, "rules_enabled": rules_enabled,
        "dictionary_sha256": dictionary_sha256(dictionary_path) if rules_enabled else None,
        "catalog_sha256": hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest() if catalog_enabled else None,
        "catalog_import": engine.catalog_report,
        **_summary(audits), "seconds": time.monotonic() - started,
        "note": "Behavior and visible-output checks are not independent validation or evidence of leaderboard improvement.",
    }
    _write_json(str(output) + ".report.json", report)
    return submission, report


def diagnostic_queries(train_path, count=32):
    """Fixed sorted keys, first source spectrum per key; never select by results."""
    keys = set()
    parquet = pq.ParquetFile(train_path)
    for batch in parquet.iter_batches(batch_size=8192, columns=["inchikey14", "ingest_lib"]):
        frame = batch.to_pandas()
        keys.update(frame.loc[frame.ingest_lib.eq("enveda-np-examples"), "inchikey14"])
    selected = sorted(keys)[:count]
    if not selected:
        raise ValueError("No enveda-np-examples diagnostic spectra available")
    pending, rows = set(selected), []
    for batch in parquet.iter_batches(batch_size=8192):
        frame = batch.to_pandas()
        subset = frame.loc[frame.ingest_lib.eq("enveda-np-examples") & frame.inchikey14.isin(pending)]
        for row in subset.drop_duplicates("inchikey14").to_dict("records"):
            key = row["inchikey14"]
            if key in pending:
                rows.append({**row, "molecule_id": key})
                pending.remove(key)
        if not pending:
            break
    return pd.DataFrame(rows).sort_values("molecule_id").reset_index(drop=True), selected


def _score_key(smiles, enumerator):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Cannot score an invalid structure")
    return Chem.MolToInchiKey(enumerator.Canonicalize(mol))[:14]


def behavior_check(train_path, coconut_path, catalog_path, dictionary_path, output,
                   *, chemistry_weight=DEFAULT_WEIGHT, count=32):
    """Real masked-reference diagnostic, explicitly not independent validation."""
    queries, selected = diagnostic_queries(train_path, count)
    neutral = queries.precursor_mz.to_numpy() - queries.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral) & (neutral > 0)]
    if not len(neutral):
        raise ValueError("Diagnostic source has no supported precursor masses")
    records, _ = load_candidates(train_path, neutral)
    selected_set = set(selected)
    # Exclude every reference spectrum of every selected key, across sources.
    records = [record for record in records if record[1] not in selected_set]
    engine = HybridChemistry(records, pd.read_parquet(coconut_path, columns=COCONUT_COLUMNS),
                             pd.read_parquet(catalog_path), load_rules(dictionary_path), chemistry_weight)
    requests = {"A_original_no_rules": (False, False), "B_expanded_no_rules": (True, False),
                "C_original_rules": (False, True), "D_expanded_rules": (True, True)}
    audits, scores = {name: [] for name in requests}, {name: [] for name in requests}
    enumerator = rdMolStandardize.TautomerEnumerator()
    for key, group in queries.groupby("molecule_id", sort=True):
        truth = _score_key(group.iloc[0].normalized_smiles, enumerator)
        for name, (pairs, audit) in engine.variants(group, requests).items():
            identities = list(dict.fromkeys(_score_key(smi, enumerator) for _, smi in pairs))
            rank = identities.index(truth) + 1 if truth in identities[:25] else 0
            scores[name].append({"key": key, "truth_identity": truth,
                                 "reciprocal_rank": 1 / rank if rank else 0,
                                 "top1": int(rank == 1), "top25": int(rank > 0)})
            audits[name].append({"molecule_id": key, **audit})
    report = {
        "status": "behavior_only_not_independent_validation",
        "source": "enveda-np-examples", "selection": "sorted first keys, first source spectrum; not selected by results",
        "requested_molecules": count, "selected_keys": selected,
        "all_selected_reference_keys_excluded": True,
        "remaining_reference_spectra": len(records),
        "chemistry_weight": chemistry_weight,
        "weight_selection": "predetermined; not tuned on these results",
        "identity": "RDKit default TautomerEnumerator.Canonicalize then InChIKey14; environment-specific diagnostic",
        "groups": {name: {**_summary(audits[name]),
                          "mrr25": float(np.mean([row["reciprocal_rank"] for row in scores[name]])),
                          "top1": float(np.mean([row["top1"] for row in scores[name]])),
                          "top25": float(np.mean([row["top25"] for row in scores[name]])),
                          "identity_results": scores[name], "audit": audits[name]}
                   for name in requests},
        "note": "Diagnostic source has been observed previously. Reference masking tests unknown-spectrum behavior; it does not establish independent generalization or score improvement. Some other-molecule references may still trigger the unchanged confidence guard.",
    }
    _write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--coconut", default="external/coconut_structures.parquet")
    parser.add_argument("--catalog", default="external/chemical_catalogs/structures.parquet")
    parser.add_argument("--dictionary", default="configs/chemical_priors.json")
    parser.add_argument("--output", default="submission_hybrid_chemistry.csv")
    parser.add_argument("--chemistry-weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--no-catalog", action="store_true", help="Original COCONUT analog pool control")
    parser.add_argument("--no-rules", action="store_true", help="Disable chemical reranking control")
    parser.add_argument("--behavior-check", action="store_true", help="Also audit four controls on 32 masked real diagnostic keys")
    args = parser.parse_args(argv)
    _, report = predict(args.data_dir, args.coconut, args.catalog, args.dictionary, args.output,
                        chemistry_weight=args.chemistry_weight,
                        catalog_enabled=not args.no_catalog, rules_enabled=not args.no_rules)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.behavior_check:
        check = behavior_check(_locate(Path(args.data_dir) / "train.parquet", "train.parquet"),
                               _locate(args.coconut, "coconut_structures.parquet"),
                               args.catalog, args.dictionary, str(args.output) + ".behavior.json",
                               chemistry_weight=args.chemistry_weight)
        print(json.dumps({"behavior_check": {name: {key: value for key, value in group.items()
                                                  if key not in {"audit", "identity_results"}}
                                            for name, group in check["groups"].items()}}, indent=2))


if __name__ == "__main__":
    main()
