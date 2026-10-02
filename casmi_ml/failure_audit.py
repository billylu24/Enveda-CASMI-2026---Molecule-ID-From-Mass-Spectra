"""Explain coverage, mass gating, rank errors, guard errors and chemical gating.

``audit_query`` consumes an existing experiment's frozen query/pool/ranking. It
does not create a split, inject truth candidates, train a model, or scan reference
spectra. Callers must exclude all raw and canonical query aliases from unknown-
spectrum references and remove held-out library structures from the candidate
pool. Public structures may legitimately contain the held-out identity.
"""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from baseline import ADDUCT_MASS
from casmi_ml.chemical_priors import extract_evidence, load_rules


ENUMERATOR = rdMolStandardize.TautomerEnumerator()
LIBRARY_ORIGINS = {"library", "train", "reference"}


@lru_cache(maxsize=250000)
def canonical_identity(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Cannot derive canonical identity from invalid SMILES")
    return Chem.MolToInchiKey(ENUMERATOR.Canonicalize(mol))[:14]


def prepare_catalog(catalog):
    """Normalize columns and compute identities once, without changing graphs/mass.

    An existing ``identity`` or ``canonical_identity`` column is reused. Input
    should be the complete frozen candidate pool, already applying its deployment
    charge policy, not merely the query's mass-filtered subset.
    """
    frame = catalog.copy()
    frame = frame.rename(columns={k: v for k, v in {
        "canonical_smiles": "normalized_smiles", "exact_mass": "mass",
    }.items() if v not in frame.columns and k in frame.columns})
    if not {"normalized_smiles", "mass", "origin"} <= set(frame):
        raise ValueError("Candidate pool requires normalized_smiles, mass and origin")
    if "canonical_identity" in frame:
        frame["identity"] = frame.canonical_identity
    elif "identity" not in frame:
        frame["identity"] = frame.normalized_smiles.map(canonical_identity)
    if frame.identity.isna().any():
        raise ValueError("Candidate identity cannot be missing")
    return frame


def _center(group):
    values = [float(r.precursor_mz) - ADDUCT_MASS.get(r.adduct, math.nan)
              for r in group.itertuples()]
    values = [v for v in values if math.isfinite(v) and v > 0]
    return float(np.median(values)) if values else None


def _normalized_mode(value):
    # Diagnostic aliases only: never alter the production matcher or ranking.
    return {"positive": "positive", "pos": "positive", "+": "positive", "1": "positive", "+1": "positive",
            "negative": "negative", "neg": "negative", "-": "negative", "-1": "negative"}.get(str(value).strip().lower())


def rule_gate_audit(group, rules, truth_smiles=None, *, ppm=10., absolute_tolerance=.002, intensity_floor=.01):
    """Run evidence extraction even for protected queries and expose every gate.

    Alias-restored matches are diagnostic only. Mass matches without a compatible
    mode/adduct are not chemical evidence and cannot authorize motif assignment.
    """
    rows = group.to_dict("records") if hasattr(group, "to_dict") else list(group)
    rules = tuple(rules)
    evidence = extract_evidence(rows, rules, ppm, absolute_tolerance, intensity_floor)
    corrected = [{**r, "ionization_mode": _normalized_mode(r.get("ionization_mode")),
                  "adduct": str(r.get("adduct", "")).strip()} for r in rows]
    alias_evidence = extract_evidence(corrected, rules, ppm, absolute_tolerance, intensity_floor)
    truth = Chem.MolFromSmiles(truth_smiles) if isinstance(truth_smiles, str) else None
    gates = []
    for rule in rules:
        counts = Counter()
        for row in rows:
            mode_ok = row.get("ionization_mode") in rule.modes
            adduct_ok = row.get("adduct") in rule.adducts
            counts["spectra"] += 1
            counts["mode_eligible"] += int(mode_ok)
            counts["adduct_eligible"] += int(adduct_ok)
            counts["joint_eligible"] += int(mode_ok and adduct_ok)
            alias_ok = (_normalized_mode(row.get("ionization_mode")) in rule.modes
                        and str(row.get("adduct", "")).strip() in rule.adducts)
            counts["alias_joint_eligible"] += int(alias_ok)
            precursor = row.get("precursor_mz")
            if precursor is None or not math.isfinite(float(precursor)) or precursor <= 0:
                counts["invalid_precursor"] += 1
                continue
            mz = np.asarray(row["ms2_mzs"], dtype=float)
            intensity = np.asarray(row["ms2_normalized_intensities"], dtype=float)
            valid = (np.isfinite(mz) & np.isfinite(intensity) & (mz > 0)
                     & (mz <= precursor + 2) & (intensity > 0))
            mz, intensity = mz[valid], intensity[valid]
            if not len(mz):
                counts["empty_valid_spectra"] += 1
                continue
            intensity /= intensity.max()
            values = mz if rule.kind == "diagnostic_ion" else precursor - mz
            tolerance = (np.full(len(mz), max(rule.mass * ppm * 1e-6, absolute_tolerance))
                         if rule.kind == "diagnostic_ion" else
                         max(precursor * ppm * 1e-6, absolute_tolerance)
                         + np.maximum(mz * ppm * 1e-6, absolute_tolerance))
            hit = np.abs(values - rule.mass) <= tolerance
            strong_hit = hit & (intensity >= intensity_floor)
            counts["mass_match_ignoring_mode_adduct"] += int(hit.any())
            if mode_ok and adduct_ok:
                counts["mass_match_joint_eligible"] += int(hit.any())
                counts["strong_match_joint_eligible"] += int(strong_hit.any())
                counts["weak_only_joint_eligible"] += int(hit.any() and not strong_hit.any())
        motif = bool(truth is not None and any(truth.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in rule.smarts))
        gates.append({"rule_id": rule.id, "truth_motif_supported": motif, **dict(counts)})
    return {
        "all_queries_evaluated_independently_of_guard": True,
        "mode_counts": dict(Counter(str(r.get("ionization_mode")) for r in rows)),
        "adduct_counts": dict(Counter(str(r.get("adduct")) for r in rows)),
        "strict_rule_matches": len(evidence), "alias_restored_rule_matches": len(alias_evidence),
        "mode_or_whitespace_encoding_issue_observed": len(alias_evidence) > len(evidence),
        "truth_supported_rule_ids": [g["rule_id"] for g in gates if g["truth_motif_supported"]],
        "rule_gates": gates, "evidence": evidence,
        "note": "Alias normalization is a diagnostic counterfactual, not a production correction or ranking change.",
    }


def audit_query(group, candidates_dataframe, baseline_ranking_pairs, rules, *, confidence=0.,
                query_truth_smiles=None, guard_threshold=.5):
    """Explain one frozen ranking against the full, caller-supplied candidate pool.

    Ranking pairs are ``(key, SMILES)`` or ``(key, SMILES, score)``. Coverage and
    rank use tautomer-canonical identities; all-query chemistry runs regardless
    of guard. A protected wrong prediction is observed, not proof that disabling
    the guard would correct it.
    """
    if group.empty:
        raise ValueError("Query group cannot be empty")
    if not math.isfinite(float(confidence)):
        raise ValueError("Confidence must be finite")
    truth_smiles = query_truth_smiles or group.iloc[0].get("normalized_smiles")
    truth = canonical_identity(truth_smiles)
    catalog = candidates_dataframe if "identity" in candidates_dataframe else prepare_catalog(candidates_dataframe)
    center = _center(group)
    delta = max(center * 35e-6, .006) if center is not None else None
    eligible = catalog[np.isfinite(catalog.mass) & catalog.mass.gt(0)
                       & catalog.mass.sub(center).abs().le(delta)] if center is not None else catalog.iloc[:0]
    overall = catalog[catalog.identity.eq(truth)]
    mass_hits = eligible[eligible.identity.eq(truth)]
    ranked = list(dict.fromkeys(canonical_identity(pair[1]) for pair in baseline_ranking_pairs))[:25]
    rank = ranked.index(truth) + 1 if truth in ranked else None
    if rank == 1:
        mechanism = "correct_top1"
    elif not len(overall):
        mechanism = "truth_absent_from_supplied_candidate_pool"
    elif center is None:
        mechanism = "unsupported_or_invalid_query_adduct"
    elif not len(mass_hits):
        mechanism = "mass_filter_miss"
    elif rank is None:
        mechanism = "ranking_or_top25_truncation"
    else:
        mechanism = "ranking_not_top1"
    protected = confidence >= guard_threshold
    origin_counts = Counter(str(x) for x in overall.origin)
    mass_origins = Counter(str(x) for x in mass_hits.origin)
    source = group.iloc[0].get("ingest_lib")
    split = group.iloc[0].get("split")
    return {
        "molecule_id": str(group.iloc[0].get("molecule_id", group.iloc[0].get("inchikey14", truth))),
        "truth_identity": truth, "query_truth_smiles": truth_smiles,
        "source": str(source) if source is not None else None,
        "split": str(split) if split is not None else None,
        "center_mass": center, "mass_window_da": delta, "mass_candidates": len(eligible),
        "truth_in_supplied_pool": bool(len(overall)), "truth_mass_eligible": bool(len(mass_hits)),
        "src_library_coverage": any(o in LIBRARY_ORIGINS for o in origin_counts),
        "src_public_coverage": any(o not in LIBRARY_ORIGINS for o in origin_counts),
        "truth_source_counts": dict(origin_counts), "truth_mass_source_counts": dict(mass_origins),
        "truth_source_masses": sorted(set(float(v) for v in overall.mass if math.isfinite(float(v)))),
        "true_rank": rank, "reciprocal_rank": 1 / rank if rank else 0.,
        "failure_mechanism": mechanism, "confidence": float(confidence), "protected": protected,
        "guard_wrong_top1": protected and rank != 1,
        "guard_protected_top25_miss": protected and rank is None,
        "chemical_audit": rule_gate_audit(group, rules, truth_smiles),
        "ranked_identities": ranked,
        "candidate_scope": "caller-supplied frozen pool; reference and split exclusions are caller responsibilities",
    }


def summarize_cases(cases):
    cases = list(cases)
    count = len(cases)
    summary = {
        "molecules": count, "mechanisms": dict(Counter(c["failure_mechanism"] for c in cases)),
        "pool_oracle_recall": sum(c["truth_in_supplied_pool"] for c in cases) / count if count else None,
        "mass_oracle_recall": sum(c["truth_mass_eligible"] for c in cases) / count if count else None,
        "mrr25": sum(c["reciprocal_rank"] for c in cases) / count if count else None,
        "top1": sum(c["true_rank"] == 1 for c in cases) / count if count else None,
        "top25": sum(c["true_rank"] is not None for c in cases) / count if count else None,
        "guard_wrong_top1": sum(c["guard_wrong_top1"] for c in cases),
        "guard_protected_top25_miss": sum(c["guard_protected_top25_miss"] for c in cases),
        "all_query_rule_matched_molecules": sum(bool(c["chemical_audit"]["strict_rule_matches"]) for c in cases),
        "truth_motif_covered_molecules": sum(bool(c["chemical_audit"]["truth_supported_rule_ids"]) for c in cases),
        "encoding_issue_molecules": sum(c["chemical_audit"]["mode_or_whitespace_encoding_issue_observed"] for c in cases),
    }
    labels = sorted({c["split"] for c in cases if c.get("split") is not None})
    if labels:
        summary["by_split"] = {label: summarize_cases([{**c, "split": None} for c in cases if c["split"] == label]) for label in labels}
    return summary


def save_audit(output, cases, *, metadata=None, representative_count=12):
    cases = list(cases)
    failures = [c for c in cases if c["failure_mechanism"] != "correct_top1"]
    report = {
        "format": "casmi_failure_audit_v1", "rdkit_version": rdkit.__version__,
        "identity": "default TautomerEnumerator.Canonicalize then InChIKey14",
        "metadata": metadata or {}, "summary": summarize_cases(cases),
        "representative_cases": failures[:representative_count], "cases": cases,
        "note": "Diagnostics do not establish independence; preserve the caller's frozen molecule/scaffold split and evaluation protocol.",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def summarize_behavior(behavior, public_catalog=None, rules=()):
    """Analyze exported score/audit pairs; never guess missing precursor/full pools."""
    groups = behavior["groups"]
    baseline = groups["A_original_no_rules"]
    expanded = groups["B_expanded_no_rules"]
    b_scores = {s["key"]: s for s in expanded["identity_results"]}
    a_audits = {a["molecule_id"]: a for a in baseline["audit"]}
    public = public_catalog.set_index("inchikey14") if public_catalog is not None else None
    cases, losses = [], []
    for score in baseline["identity_results"]:
        key = score["key"]
        audit = a_audits[key]
        row = public.loc[key] if public is not None and key in public.index else None
        old_rr, new_rr = score["reciprocal_rank"], b_scores[key]["reciprocal_rank"]
        case = {"key": key, "truth_identity": score["truth_identity"], "confidence": audit["confidence"],
                "protected": audit["protected"], "old_rank": round(1 / old_rr) if old_rr else None,
                "expanded_rank": round(1 / new_rr) if new_rr else None,
                "public_rawkey_covered": row is not None, "public_charge": int(row.formal_charge) if row is not None else None,
                "motif_rules_in_public_representative": [], "public_source_name": None,
                "failure": "present_in_output_but_not_top1" if old_rr else "top25_miss_cause_unresolved"}
        if row is not None:
            mol = Chem.MolFromSmiles(row.normalized_smiles)
            case["motif_rules_in_public_representative"] = [rule.id for rule in rules if any(mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in rule.smarts)]
            provenance = json.loads(row.provenance_json) if "provenance_json" in row else []
            case["public_source_name"] = next((p.get("annotations", {}).get("ChEBI NAME") or p.get("annotations", {}).get("NAME") for p in provenance if p.get("annotations")), None)
        if new_rr < old_rr:
            losses.append(case)
        cases.append(case)
    return {
        "status": "existing_behavior_only_not_independent_validation",
        "molecules": len(cases), "protected_molecules": sum(c["protected"] for c in cases),
        "protected_wrong_top1": sum(c["protected"] and c["old_rank"] != 1 for c in cases),
        "protected_top25_miss": sum(c["protected"] and c["old_rank"] is None for c in cases),
        "unprotected_molecules_actually_eligible_for_rule_extraction": sum(not c["protected"] for c in cases),
        "public_rawkey_covered": sum(c["public_rawkey_covered"] for c in cases),
        "public_representative_motif_covered": sum(bool(c["motif_rules_in_public_representative"]) for c in cases),
        "expanded_rr_losses": losses, "cases": cases,
        "limitations": ["No per-query precursor, ion mode, adduct, peaks, full candidate pool or library score margin was exported.",
                        "A Top25 miss cannot separate missing eligible candidates from mass gating or ranking truncation.",
                        "Protected queries skipped extraction; recorded zero rules is not an all-query spectral coverage measurement.",
                        "Public matching representatives are not recovered original query graphs; raw-key coverage is not a mass-eligibility claim."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    behavior = commands.add_parser("outputs", help="audit existing exported behavior results")
    behavior.add_argument("--behavior", type=Path, required=True)
    behavior.add_argument("--catalog", type=Path)
    behavior.add_argument("--dictionary", type=Path, default=Path("configs/chemical_priors.json"))
    behavior.add_argument("--output", type=Path, required=True)
    query = commands.add_parser("queries", help="audit a frozen holdout and supplied model rankings")
    query.add_argument("--queries", type=Path, required=True)
    query.add_argument("--pool", type=Path, required=True, help="full already-masked unified library/public candidate pool")
    query.add_argument("--rankings", type=Path, required=True, help="JSON list: molecule_id, pairs of key/SMILES, confidence")
    query.add_argument("--dictionary", type=Path, default=Path("configs/chemical_priors.json"))
    query.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rules = load_rules(args.dictionary)
    if args.command == "outputs":
        report = summarize_behavior(json.loads(args.behavior.read_text()), pd.read_parquet(args.catalog) if args.catalog else None, rules)
        report["behavior_sha256"] = hashlib.sha256(args.behavior.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    else:
        frame, pool = pd.read_parquet(args.queries), prepare_catalog(pd.read_parquet(args.pool))
        rankings = {str(r["molecule_id"]): r for r in json.loads(args.rankings.read_text())}
        key = "molecule_id" if "molecule_id" in frame else "inchikey14"
        cases = []
        for molecule, group in frame.groupby(key, sort=True):
            ranking = rankings[str(molecule)]
            cases.append(audit_query(group, pool, ranking["pairs"], rules, confidence=ranking.get("confidence", 0.)))
        report = save_audit(args.output, cases, metadata={"queries": args.queries.name, "pool": args.pool.name,
                                                       "rankings": args.rankings.name, "split_policy": "reuse supplied frozen split; no new split or tuning"})
    print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "representative_cases", "expanded_rr_losses"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
