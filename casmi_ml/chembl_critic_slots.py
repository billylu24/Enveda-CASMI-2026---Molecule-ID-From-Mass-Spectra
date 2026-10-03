"""Critic margin gates bounded native ChEMBL proposals, with0062 ranks frozen."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import fingerprint, write_json
from casmi_ml.direct_models import DirectRanker, score_group
from casmi_ml.generated_first_critic import CRITIC, promotion_prefix
from casmi_ml.generated_score_combination import (
    GENERATED,
    SCORES,
    SOURCE,
    combined_order,
)
from casmi_ml.generated_second_reference import select_prefix
from casmi_ml.generation_slots import insert_generated
from casmi_ml.mass_candidates import mass_centers
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex, metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

PROPOSALS = Path(
    "artifacts/research_loop/rounds/0071_chembl_candidate_slots/proposals.json"
)

VARIANTS = {
    "baseline": None,
    "margin0_prefix2": (0.0, 2, 3),
    "margin005_prefix2": (0.05, 2, 3),
    "margin005_prefix5": (0.05, 5, 3),
    "margin005_prefix10": (0.05, 10, 3),
}


def score_cache_key(query, first, candidates, model_binding=None):
    values = [query, first, candidates]
    if model_binding is not None:
        values.append(model_binding)
    payload = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_proposal_membership(proposals, original):
    if set(proposals) != set(original) or any(
        len(values) != len(set(values)) or set(values) != set(original[key])
        for key, values in proposals.items()
    ):
        raise ValueError("Custom proposals must preserve each original mass window")


def validate_all_query_mass_windows(proposals, index, frame, groups):
    if set(proposals) != set(groups):
        raise ValueError("All-query proposals require every frozen query key")
    for key, ids in groups.items():
        query = frame.iloc[ids][["adduct", "precursor_mz"]]
        allowed = {
            candidate
            for center in mass_centers(query, "charge_aware_union")
            for candidate in index.query(center).inchikey14
        }
        if (
            len(proposals[key]) != len(set(proposals[key]))
            or set(proposals[key]) != allowed
        ):
            raise ValueError("All-query proposals differ from observable mass window")


@torch.inference_mode()
def run(
    output,
    incumbent,
    proposal_limit=100,
    critic_checkpoint=CRITIC,
    proposal_encoder=ENCODER,
    proposal_path=PROPOSALS,
    confidence_scope="low",
):
    if confidence_scope not in ("low", "high"):
        raise ValueError("Confidence scope must be low or high")
    variants = (
        VARIANTS
        if confidence_scope == "low"
        else {"baseline": None, "margin005_prefix10": VARIANTS["margin005_prefix10"]}
    )
    proposal_path = Path(proposal_path)
    critic_checkpoint, proposal_encoder = (
        Path(critic_checkpoint),
        Path(proposal_encoder),
    )
    encoder_sha = digest(proposal_encoder)
    critic_sha = digest(critic_checkpoint)
    model_binding = {"encoder_sha256": encoder_sha, "critic_sha256": critic_sha}
    if proposal_limit not in (100, 500):
        raise ValueError("Proposal limit must be100 or500")
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    pair_path = Path(
        "artifacts/research_loop/rounds/0061_generated_first_critic/scores.json"
    )
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "catalog_sha256": digest(DERIVED),
            "encoder_sha256": encoder_sha,
            "baseline_encoder_sha256": digest(ENCODER),
            "encoder_scope": "External proposals and actual first critic only;0062 retrieval and generation frozen",
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "promotion_scores_sha256": digest(pair_path),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": variants,
            "confidence_scope": confidence_scope,
            "high_scope_rule": "Only confidence>=.5, preserve original first10, at most3 novel proposals after10, actual first critic+.05 gate"
            if confidence_scope == "high"
            else None,
            "proposal_limit": proposal_limit,
            "rule": f"Freeze0062;confidence<.5;preselect first{proposal_limit} novel ChEMBL proposals by native fingerprint;critic ranks proposals and top proposal must exceed actual current first by margin; insert at most3 after prefix",
            "critic_sha256": critic_sha,
            "native_proposals_sha256": digest(proposal_path),
            "proposal_source_protocol_sha256": digest(
                proposal_path.parent / "protocol.json"
            ),
            "proposal_path": str(proposal_path),
            "mass_hypothesis": "charge_aware_union",
            "new_training": False,
            "new_sampling": False,
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "cohort": "repeated_development",
            "score_seconds": 3600,
        },
    )
    configure(42, threads=4)
    manifest = json.loads(Path("external/chembl37/manifest.json").read_text())
    if digest(DERIVED) != manifest["derived_sha256"]:
        raise ValueError("External source changed")
    index = CandidateIndex(pd.read_parquet(DERIVED))
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    generated = json.loads(GENERATED.read_text())
    scores = json.loads(SCORES.read_text())
    ordered = {
        r["key"]: combined_order(
            r["candidates"],
            scores.get(r["key"], {}).get("fingerprint", {}),
            (1, 0, 0.5),
        )
        for r in generated
    }
    pairs = json.loads(pair_path.read_text())
    if len(generated) != 2000 or set(ordered) != set(groups):
        raise ValueError("Full2000 keys required")
    encoder, saved = load_deployment_checkpoint(proposal_encoder, "scale")
    encoder.eval()
    proposal_protocol = json.loads((proposal_path.parent / "protocol.json").read_text())
    if proposal_protocol["catalog_sha256"] != digest(DERIVED) or proposal_protocol[
        "encoder_sha256"
    ] != digest(ENCODER):
        raise ValueError("Proposal source catalog/base encoder differs")
    proposals = json.loads(proposal_path.read_text())
    if proposal_path != PROPOSALS:
        if proposal_protocol.get("development_sha256") != digest(
            ROOT / "researchdev.parquet"
        ):
            raise ValueError("Custom proposals require explicit cohort binding")
        if proposal_protocol.get("all_queries"):
            validate_all_query_mass_windows(proposals, index, frame, groups)
        else:
            validate_proposal_membership(proposals, json.loads(PROPOSALS.read_text()))
    if confidence_scope == "high" and not proposal_protocol.get("all_queries"):
        raise ValueError(
            "High-confidence scoring requires complete all-query proposals"
        )
    catalog_keys = set(index.catalog.inchikey14)
    if not set(proposals).issubset(groups) or any(
        len(values) != len(set(values)) or not set(values).issubset(catalog_keys)
        for values in proposals.values()
    ):
        raise ValueError("Invalid proposal identities")
    del catalog_keys
    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    lookup.update(
        {
            r.inchikey14: r.normalized_smiles
            for r in pd.read_parquet(
                "external/pubchemlite/structures.parquet"
            ).itertuples()
            if r.inchikey14 not in lookup
        }
    )
    external_lookup = index.catalog.set_index("inchikey14").normalized_smiles.to_dict()
    generated_lookup = {
        r["key"]: {c["key"]: c["smiles"] for c in r["candidates"]} for r in generated
    }
    weights = torch.load(critic_checkpoint, map_location="cpu", weights_only=True)
    if (
        weights["encoder_sha256"] != encoder_sha
        or weights["architecture"] != "fingerprint"
    ):
        raise ValueError("Proposal critic/encoder binding differs")
    critic = DirectRanker("fingerprint").eval()
    critic.load_state_dict(weights["state_dict"])
    cache_path = output / "critic_scores.json"
    pair_scores = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    budget = StageBudget(
        output,
        "catalog_cpu",
        "proposal_scoring",
        3600,
        limit=3600,
        lock_path=output / "score.lock",
    )
    started = time.monotonic()
    report, diagnostics, coverage = {}, {}, {}
    try:
        for mode in ("unknown", "known"):
            rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
            confidence = {
                r["key"]: r["confidence"]
                for r in json.loads(
                    (ROOT / f"researchdev_{mode}_chemical.json").read_text()
                )
            }
            observed = set(
                pd.read_parquet(
                    Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                    / mode
                    / "rows.parquet",
                    columns=["inchikey14"],
                ).inchikey14
            )
            if {r["key"] for r in rows} != set(groups):
                raise ValueError("Retrieval keys differ")
            if mode == "known":
                lookup.update(candidate_lookup(ROOT, "researchdev", "known"))
            coverage[mode] = {
                "queries": 0,
                "native_contains_truth": 0,
                "critic_top3": 0,
                "critic_top10": 0,
                "critic_top25": 0,
            }
            ranks = {name: {} for name in variants}
            pools = {name: {} for name in variants}
            diagnostics[mode] = {
                name: {
                    "queries_with_inserted_candidates": 0,
                    "novel_inserted_truths": 0,
                }
                for name in variants
            }
            for i, row in enumerate(rows, 1):
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                protected = protects_reference(
                    base["ranking"], observed, confidence[key]
                )
                selected = base if protected else row["variants"]["expansion_1"]
                prefix = (
                    select_prefix(
                        base["ranking"],
                        observed,
                        confidence[key],
                        "second_unreferenced",
                    )
                    if protected
                    else 2
                )
                candidates = ordered[key]
                novel = [k for k in candidates if k not in set(selected["ranking"])]
                if confidence[key] < 0.2 and selected["ranking"] and novel:
                    prefix = promotion_prefix(
                        selected["ranking"],
                        candidates,
                        pairs[key + ":" + selected["ranking"][0] + ":" + novel[0]],
                        prefix,
                        0.05,
                    )
                current = insert_generated(selected["ranking"], candidates, prefix, 5)
                current_pool = selected["pool"] + candidates
                candidates_external = []
                cache_key = None
                in_scope = (
                    confidence[key] < 0.5
                    if confidence_scope == "low"
                    else confidence[key] >= 0.5
                )
                if in_scope and current:
                    current_set = set(current)
                    candidates_external = [
                        k for k in proposals[key] if k not in current_set
                    ][:proposal_limit]
                    cache_key = score_cache_key(
                        key, current[0], candidates_external, model_binding
                    )
                    if candidates_external and cache_key not in pair_scores:
                        if not budget.checkpoint():
                            raise TimeoutError("Proposal critic budget exhausted")
                        query = frame.iloc[groups[key]].drop(
                            columns=[
                                c
                                for c in [
                                    "inchikey14",
                                    "normalized_smiles",
                                    "molecular_formula",
                                    "fingerprint",
                                ]
                                if c in frame
                            ]
                        )
                        first_smiles = generated_lookup[key].get(
                            current[0], lookup.get(current[0])
                        )
                        if first_smiles is None:
                            raise ValueError(
                                "Current first candidate representation missing"
                            )
                        smiles = [first_smiles] + [
                            external_lookup[k] for k in candidates_external
                        ]
                        values = score_group(
                            critic,
                            encoder,
                            query,
                            saved["preprocessing"],
                            pd.DataFrame({"normalized_smiles": smiles}),
                            np.stack([fingerprint(s) for s in smiles]).astype(
                                np.float32
                            ),
                        )
                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite catalog critic scores")
                        pair_scores[cache_key] = {
                            k: float(v)
                            for k, v in zip([current[0]] + candidates_external, values)
                        }
                        if len(pair_scores) % 25 == 0:
                            write_json(cache_path, pair_scores)
                            print(
                                "chembl_critic",
                                len(pair_scores),
                                "seconds",
                                time.monotonic() - started,
                                flush=True,
                            )
                    if candidates_external:
                        candidates_external = sorted(
                            candidates_external,
                            key=lambda k: (-pair_scores[cache_key][k], k),
                        )
                if candidates_external:
                    coverage[mode]["queries"] += 1
                    coverage[mode]["native_contains_truth"] += (
                        key in candidates_external
                    )
                    for cutoff in (3, 10, 25):
                        coverage[mode][f"critic_top{cutoff}"] += (
                            key in candidates_external[:cutoff]
                        )
                for name, spec in variants.items():
                    external = []
                    if (
                        spec
                        and candidates_external
                        and pair_scores[cache_key][candidates_external[0]]
                        > pair_scores[cache_key][current[0]] + spec[0]
                    ):
                        external = candidates_external
                    novel_external = (
                        [k for k in external if k not in set(current)][: spec[2]]
                        if spec
                        else []
                    )
                    ranks[name][key] = (
                        insert_generated(current, external, spec[1], spec[2])
                        if external
                        else current
                    )
                    pools[name][key] = current_pool + external
                    diagnostics[mode][name]["queries_with_inserted_candidates"] += bool(
                        novel_external
                    )
                    diagnostics[mode][name]["novel_inserted_truths"] += (
                        key in novel_external
                    )
            report[mode] = {}
            for name in variants:
                result, per = metrics(ranks[name], pools[name])
                if name == "baseline":
                    expected = (
                        pd.read_csv(incumbent / f"{mode}_confidence0.2_margin0.05.csv")
                        .set_index("key")
                        .sort_index()
                    )
                    actual = per.set_index("key").sort_index()
                    if (
                        not actual.index.equals(expected.index)
                        or (
                            actual[["reciprocal_rank", "top1"]]
                            - expected[["reciprocal_rank", "top1"]]
                        )
                        .abs()
                        .max()
                        .max()
                        > 1e-12
                    ):
                        raise ValueError("Paired baseline must match0062")
                report[mode][name] = result
                per.to_csv(output / f"{mode}_{name}.csv", index=False)
        write_json(cache_path, pair_scores)
        write_json(output / "diagnostics.json", diagnostics)
        report["diagnostics"] = {
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "proposal_queries_scored": len(pair_scores),
            "proposal_limit": proposal_limit,
            "proposal_coverage": coverage,
            "variants": diagnostics,
            "independent_acceptance": False,
        }
        write_json(output / "report.json", report)
        return report
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--proposal-limit", type=int, choices=(100, 500), default=100)
    p.add_argument("--critic-checkpoint", type=Path, default=CRITIC)
    p.add_argument("--proposal-encoder", type=Path, default=ENCODER)
    p.add_argument("--proposal-path", type=Path, default=PROPOSALS)
    p.add_argument("--confidence-scope", choices=("low", "high"), default="low")
    a = p.parse_args()
    print(
        json.dumps(
            run(
                a.output,
                a.incumbent,
                a.proposal_limit,
                a.critic_checkpoint,
                a.proposal_encoder,
                a.proposal_path,
                a.confidence_scope,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
