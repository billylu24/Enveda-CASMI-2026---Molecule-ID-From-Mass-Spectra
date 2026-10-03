"""Fixed200 pilot: real fragmentation reranks critic-gated ChEMBL proposals."""

import argparse
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chemistry import extract_evidence, rerank, rule_manifest
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
from casmi_ml.metfrag import MetFrag, digest
from casmi_ml.metfrag import score_group as fragment_score_group
from casmi_ml.ranking import CandidateIndex, metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

VARIANTS = {
    "baseline": None,
    "critic_control": (0.05, 5, 3, 0.0),
    "fragment05": (0.05, 5, 3, 0.5),
    "fragment1": (0.05, 5, 3, 1.0),
}
JAVA = "external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java"


def informative_fragments(proposed, fragments):
    values = np.asarray([fragments.get(k, 0.0) for k in proposed], dtype=float)
    return bool(
        len(values) > 1
        and np.isfinite(values).all()
        and values.max() > 0
        and np.ptp(values) > 0
        and values[0] > 0
    )


def possible_critic_gate(candidates, scores, current_first, margin):
    return bool(
        candidates
        and max(scores[k] for k in candidates) > scores[current_first] + margin
    )


def supported_proposals(proposed, critic_scores, current_first, fragments, margin):
    return [
        k
        for k in proposed
        if fragments.get(k, 0.0) > 0
        and critic_scores[k] > critic_scores[current_first] + margin
    ]


def relative_supported_proposals(proposed, fragments, current_first):
    threshold = max(0.0, fragments.get(current_first, 0.0))
    return [
        k
        for k in proposed
        if np.isfinite(fragments.get(k, 0.0)) and fragments.get(k, 0.0) > threshold
    ]


def score_cache_key(query, first, candidates):
    payload = json.dumps([query, first, candidates], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@torch.inference_mode()
def run(
    output,
    incumbent,
    limit=200,
    prefix=5,
    monomer=False,
    proposal_limit=100,
    fragment_limit=100,
    evidence_only=False,
    critic_checkpoint=CRITIC,
    dimer=False,
    candidate_gate=False,
    compare_first=False,
    fragment_gate_only=False,
    skip_impossible=False,
    relative_candidate_gate=False,
    fragment_weight=0.5,
    mean_fragments=False,
    chemical_prior=False,
    fragment_depth=2,
    sequence_scores=None,
    sequence_ratio=1.0,
    fingerprint_prior_scores=None,
    critic_only=False,
    pilot_selector="prefix",
):
    if pilot_selector not in ("prefix", "hash"):
        raise ValueError("Pilot selector must be prefix or hash")
    if critic_only and (
        evidence_only
        or compare_first
        or mean_fragments
        or relative_candidate_gate
        or chemical_prior
    ):
        raise ValueError(
            "Critic-only control cannot require fragment evidence or chemical-prior gates"
        )
    if sequence_scores is not None and fingerprint_prior_scores is not None:
        raise ValueError(
            "Sequence and fingerprint-prior scoring are isolated experiments"
        )
    if sequence_ratio not in (0.0, 1.0):
        raise ValueError("Sequence ratio weight must be 0 or 1")
    if type(fragment_depth) is not int or fragment_depth not in (2, 3):
        raise ValueError("Fragment depth must be 2 or 3")
    if fragment_weight not in (0.5, 1.0):
        raise ValueError("Fragment weight must be 0.5 or 1.0")
    if relative_candidate_gate and (
        not compare_first or not evidence_only or candidate_gate
    ):
        raise ValueError(
            "Relative candidate gate requires first comparison and informative evidence, without per-candidate critic gate"
        )
    if skip_impossible and fragment_gate_only:
        raise ValueError("Cannot skip critic gate when that gate is disabled")
    if fragment_gate_only and (
        not compare_first or not evidence_only or candidate_gate
    ):
        raise ValueError(
            "Fragment-only gate requires first comparison, informative evidence and no candidate critic gate"
        )
    if compare_first and not evidence_only:
        raise ValueError("Fragment first comparison requires informative evidence")
    if candidate_gate and not evidence_only:
        raise ValueError("Candidate gate requires informative fragment evidence")
    if dimer and not monomer:
        raise ValueError("Dimer experiment includes monomer adapter")
    adapter_name = (
        "observed_dimer_monomer_products"
        if dimer
        else "exact_monomer"
        if monomer
        else "proton_only"
    )
    if mean_fragments and not compare_first:
        raise ValueError(
            "Mean fragment evidence requires actual first in the same spectra"
        )
    execution_engine_sha256 = digest("casmi_ml/metfrag.py")
    execution_aggregation_sha256 = (
        digest("casmi_ml/paired_fragments.py")
        if mean_fragments
        else execution_engine_sha256
    )
    critic_checkpoint = Path(critic_checkpoint)
    if proposal_limit not in (100, 500) or not 1 <= fragment_limit <= proposal_limit:
        raise ValueError("Invalid proposal or fragment shortlist limit")
    critic_path = Path(
        "artifacts/research_loop/rounds/0078_chembl_critic_wide/critic_scores.json"
        if proposal_limit == 500
        else "artifacts/research_loop/rounds/0073_chembl_critic_slots/critic_scores.json"
    )
    if prefix not in (3, 5, 10):
        raise ValueError("Insertion prefix must be3,5 or10")
    variants = {
        name: None if spec is None else (spec[0], prefix, *spec[2:])
        for name, spec in VARIANTS.items()
    }
    if evidence_only:
        variants = {
            "baseline": None,
            "fragment05": (0.05, 2, 3, 0.5),
            "fragment05_prefix3": (0.05, 3, 3, 0.5),
            "fragment05_prefix5": (0.05, 5, 3, 0.5),
            "fragment05_prefix10": (0.05, 10, 3, 0.5),
        }
    if evidence_only:
        variants = {
            name: None if spec is None else (*spec[:3], fragment_weight)
            for name, spec in variants.items()
        }
    if limit == 2000:
        variants = {
            name: spec for name, spec in variants.items() if name != "fragment1"
        }
    if evidence_only and fragment_weight == 1.0:
        variants = {
            name.replace("fragment05", "fragment1"): spec
            for name, spec in variants.items()
        }
    if limit == 2000 and evidence_only:
        variants = {
            "baseline": None,
            f"{'fragment1' if fragment_weight == 1.0 else 'fragment05'}_prefix{prefix}": (
                0.05,
                prefix,
                3,
                fragment_weight,
            ),
        }
    if critic_only:
        variants = {"baseline": None, "critic_control": (0.05, prefix, 3, 0.0)}
    if not 1 <= limit <= 2000:
        raise ValueError("Pilot limit must be in[1,2000]")
    sequence_values = {}
    if sequence_scores is not None:
        sequence_scores = Path(sequence_scores)
        sequence_protocol = json.loads(
            (sequence_scores.parent / "protocol.json").read_text()
        )
        sequence_report = json.loads(
            (sequence_scores.parent / "report.json").read_text()
        )
        if (
            sequence_protocol["encoder_sha256"] != digest(ENCODER)
            or sequence_protocol["catalog_sha256"] != digest(DERIVED)
            or sequence_protocol["development_sha256"]
            != digest(ROOT / "researchdev.parquet")
            or sequence_protocol["limit"] != 2000
            or not sequence_report["diagnostic_only"]
            or sequence_report["external_only"]["native"]["molecules"] != 2000
        ):
            raise ValueError(
                "Require frozen full2000 external sequence scores bound to this encoder/catalog/cohort"
            )
        sequence_values = json.loads(sequence_scores.read_text())
    prior_values = {}
    if fingerprint_prior_scores is not None:
        fingerprint_prior_scores = Path(fingerprint_prior_scores)
        prior_protocol = json.loads(
            (fingerprint_prior_scores.parent / "protocol.json").read_text()
        )
        prior_report = json.loads(
            (fingerprint_prior_scores.parent / "report.json").read_text()
        )
        if (
            prior_protocol["encoder_sha256"] != digest(ENCODER)
            or prior_protocol["catalog_sha256"] != digest(DERIVED)
            or prior_protocol["development_sha256"]
            != digest(ROOT / "researchdev.parquet")
            or prior_protocol["limit"] != 2000
            or prior_report["external_only"]["native"]["molecules"] != 2000
        ):
            raise ValueError(
                "Require frozen full2000 training-prior scores with matching encoder/catalog/cohort"
            )
        prior_values = json.loads(fingerprint_prior_scores.read_text())
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    seed_path = output / "initial_fragment_scores.json"
    if not seed_path.exists():
        if (output / "fragment_scores.json").exists():
            shutil.copyfile(output / "fragment_scores.json", seed_path)
        else:
            write_json(seed_path, {})
    pair_path = Path(
        "artifacts/research_loop/rounds/0061_generated_first_critic/scores.json"
    )
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "catalog_sha256": digest(DERIVED),
            "encoder_sha256": digest(ENCODER),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "promotion_scores_sha256": digest(pair_path),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": variants,
            "rule": f"Freeze0062;confidence<.5;native first{proposal_limit} novel ChEMBL proposals;critic shortlist first{fragment_limit}, then MetFrag tie-aware rerank; actual first proposal critic must exceed current first by.05;insert3 after frozen prefix",
            "limit": limit,
            "pilot_selector": pilot_selector,
            "pilot_selector_salt": "fragment-representative-20261003"
            if pilot_selector == "hash"
            else None,
            "insertion_prefix": prefix,
            "fingerprint_prior_ranking": fingerprint_prior_scores is not None,
            "sequence_ranking": sequence_scores is not None,
            "proposal_limit": proposal_limit,
            "fragment_limit": fragment_limit,
            "initial_fragment_cache_sha256": digest(seed_path),
            "evidence_only": evidence_only,
            "critic_only": critic_only,
            "candidate_gate": candidate_gate,
            "relative_candidate_gate": relative_candidate_gate,
            "chemical_prior": chemical_prior,
            "fragment_weight": fragment_weight,
            "fragment_aggregation": "mean_normalized" if mean_fragments else "max_raw",
            "fragment_aggregation_source_sha256": execution_aggregation_sha256,
            "compare_current_first_fragment": compare_first,
            "fragment_gate_only": fragment_gate_only,
            "skip_impossible_critic_gate": skip_impossible,
            "evidence_gate": "Positive proposed first fragment score and distinct finite fragment scores required; missing/all tied scores never enable insertion"
            if evidence_only
            else None,
            "proposal_critic_cache_sha256": digest(critic_path)
            if critic_checkpoint == CRITIC
            else None,
            "fragment_depth": fragment_depth,
            "fragment_engine_source_sha256": execution_engine_sha256,
            "fragment_adapter": adapter_name,
            "fragment_adapter_sha256": digest(
                "casmi_ml/metfrag_dimer.py"
                if dimer
                else "casmi_ml/metfrag_monomer.py"
                if monomer
                else "casmi_ml/metfrag.py"
            ),
            "fragment_seconds": 1200,
            "java_sha256": digest(JAVA),
            "jar_sha256": digest("external/metfrag/MetFragCommandLine-2.6.11.jar"),
            "critic_sha256": digest(critic_checkpoint),
            "native_proposals_sha256": digest(
                Path(
                    "artifacts/research_loop/rounds/0071_chembl_candidate_slots/proposals.json"
                )
            ),
            "mass_hypothesis": "charge_aware_union",
            "chemical_prior_weight": 0.25 if chemical_prior else 0.0,
            "chemical_prior_rules": rule_manifest() if chemical_prior else None,
            "chemistry_source_sha256": digest("casmi_ml/chemistry.py"),
            "new_training": False,
            "fingerprint_prior_scores_sha256": digest(fingerprint_prior_scores)
            if fingerprint_prior_scores
            else None,
            "fingerprint_prior_protocol_sha256": digest(
                fingerprint_prior_scores.parent / "protocol.json"
            )
            if fingerprint_prior_scores
            else None,
            "fingerprint_prior_fusion_weight": 0.5
            if fingerprint_prior_scores
            else None,
            "sequence_scores_sha256": digest(sequence_scores)
            if sequence_scores
            else None,
            "sequence_protocol_sha256": digest(sequence_scores.parent / "protocol.json")
            if sequence_scores
            else None,
            "sequence_ratio_weight": sequence_ratio if sequence_scores else None,
            "sequence_fusion_weight": 0.5 if sequence_scores else None,
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
    encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
    encoder.eval()
    proposals = json.loads(
        Path(
            "artifacts/research_loop/rounds/0071_chembl_candidate_slots/proposals.json"
        ).read_text()
    )
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
    assert (
        weights["encoder_sha256"] == digest(ENCODER)
        and weights["architecture"] == "fingerprint"
    )
    critic = DirectRanker("fingerprint").eval()
    critic.load_state_dict(weights["state_dict"])
    cache_path = output / "critic_scores.json"
    pair_scores = (
        json.loads(cache_path.read_text())
        if cache_path.exists()
        else json.loads(critic_path.read_text())
        if critic_checkpoint == CRITIC
        else {}
    )
    selection_keys = sorted(groups)
    if pilot_selector == "hash":
        selection_keys.sort(
            key=lambda k: hashlib.sha256(
                ("fragment-representative-20261003:" + k).encode()
            ).digest()
        )
    allowed_keys = set(selection_keys[:limit])
    adapter = MetFrag
    if monomer:
        from casmi_ml.metfrag_monomer import MonomerMetFrag

        adapter = MonomerMetFrag
    if dimer:
        from casmi_ml.metfrag_dimer import DimerMetFrag

        adapter = DimerMetFrag
    fragmenter = adapter(
        "external/metfrag/MetFragCommandLine-2.6.11.jar",
        ROOT / "chembl_metfrag_cache",
        java=JAVA,
        depth=fragment_depth,
    )
    group_scorer = fragment_score_group
    if mean_fragments:
        from casmi_ml.paired_fragments import score_group_mean

        group_scorer = score_group_mean
    fragment_path = output / "fragment_scores.json"
    fragment_scores = (
        json.loads(fragment_path.read_text()) if fragment_path.exists() else {}
    )
    budget = StageBudget(
        output,
        "catalog_cpu",
        "proposal_scoring",
        3600,
        limit=3600,
        lock_path=output / "score.lock",
    )
    started = time.monotonic()
    report, diagnostics = {}, {}
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
                if row["key"] not in allowed_keys or (
                    mode == "known" and not row["known"]
                ):
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
                if confidence[key] < 0.5 and current:
                    current_set = set(current)
                    candidates_external = [
                        k for k in proposals[key] if k not in current_set
                    ][:proposal_limit]
                    cache_key = score_cache_key(key, current[0], candidates_external)
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
                        if sequence_scores is not None:
                            if key not in sequence_values:
                                raise ValueError(
                                    "Full external sequence cache missing a query with proposals"
                                )
                            from casmi_ml.chembl_sequence_pilot import reorder_scoreable

                            candidates_external = reorder_scoreable(
                                candidates_external,
                                sequence_values.get(key, {}),
                                sequence_ratio,
                            )
                        if fingerprint_prior_scores is not None:
                            from casmi_ml.chembl_sequence_pilot import reorder_scoreable

                            mapped_scores = {
                                k: {"query": v, "prior": 0.0}
                                for k, v in prior_values.get(key, {}).items()
                            }
                            candidates_external = reorder_scoreable(
                                candidates_external, mapped_scores, 0.0
                            )
                        candidates_external = candidates_external[:fragment_limit]
                if (
                    skip_impossible
                    and candidates_external
                    and not possible_critic_gate(
                        candidates_external,
                        pair_scores[cache_key],
                        current[0],
                        min(spec[0] for spec in variants.values() if spec),
                    )
                ):
                    candidates_external = []
                fragment_key = cache_key
                if compare_first and candidates_external:
                    fragment_key = score_cache_key(
                        "include_current_first_v1:" + key,
                        current[0],
                        candidates_external,
                    )
                if mean_fragments and candidates_external:
                    fragment_key = score_cache_key(
                        "mean_normalized_v1:" + key, current[0], candidates_external
                    )
                if fragment_depth != 2 and candidates_external:
                    fragment_key = score_cache_key(
                        f"depth{fragment_depth}:" + str(fragment_key),
                        current[0],
                        candidates_external,
                    )
                fragments, fallback = {}, False
                if candidates_external and not critic_only:
                    if fragment_key not in fragment_scores:
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
                        fragment_candidates = {
                            k: external_lookup[k] for k in candidates_external
                        }
                        if compare_first:
                            first_smiles = generated_lookup[key].get(
                                current[0], lookup.get(current[0])
                            )
                            if first_smiles is None:
                                raise ValueError("Current first representation missing")
                            fragment_candidates[current[0]] = first_smiles
                        fragments, fallback = group_scorer(
                            fragmenter,
                            query.to_dict("records"),
                            fragment_candidates,
                            deadline=budget.started + 1200,
                        )
                        fragment_scores[fragment_key] = {
                            "scores": fragments,
                            "budget_fallback": fallback,
                        }
                        write_json(fragment_path, fragment_scores)
                        print(
                            "catalog_fragment_scored",
                            len(fragment_scores),
                            "seconds",
                            time.monotonic() - started,
                            flush=True,
                        )
                    fragments = fragment_scores[fragment_key]["scores"]
                    fallback = fragment_scores[fragment_key]["budget_fallback"]
                for name, spec in variants.items():
                    proposed = (
                        rerank(
                            candidates_external,
                            {},
                            [],
                            spec[3],
                            top_n=max(1, len(candidates_external)),
                            fragment_scores=fragments,
                        )
                        if spec
                        else []
                    )
                    if chemical_prior and proposed:
                        records = (
                            frame.iloc[groups[key]]
                            .drop(
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
                            .to_dict("records")
                        )
                        proposed = rerank(
                            proposed,
                            external_lookup,
                            [extract_evidence(r) for r in records],
                            0.25,
                            top_n=max(1, len(proposed)),
                        )
                    external = []
                    if (
                        spec
                        and proposed
                        and not (spec[3] and fallback)
                        and (
                            not compare_first
                            or fragments.get(proposed[0], 0.0)
                            > fragments.get(current[0], 0.0)
                        )
                        and (
                            not evidence_only
                            or informative_fragments(proposed, fragments)
                        )
                        and (
                            fragment_gate_only
                            or pair_scores[cache_key][proposed[0]]
                            > pair_scores[cache_key][current[0]] + spec[0]
                        )
                    ):
                        external = (
                            supported_proposals(
                                proposed,
                                pair_scores[cache_key],
                                current[0],
                                fragments,
                                spec[0],
                            )
                            if candidate_gate
                            else relative_supported_proposals(
                                proposed, fragments, current[0]
                            )
                            if relative_candidate_gate
                            else proposed
                        )
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
                    expected = expected.loc[actual.index]
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
        report["diagnostic_only"] = limit < 2000
        report["diagnostics"] = {
            "molecules": len(allowed_keys),
            "incremental_spectrum_execution_status_counts": getattr(
                fragmenter, "status_counts", {}
            ),
            "execution_status_scope": "Only spectra actually traversed by this run; precomputed group cache skips are excluded. Complete may be content-cache reuse, not cold Java execution.",
            "fragment_groups": len(fragment_scores),
            "fragment_groups_with_scores": sum(
                bool(r["scores"]) for r in fragment_scores.values()
            ),
            "fragment_depth": fragment_depth,
            "fragment_engine_source_sha256": execution_engine_sha256,
            "fragment_adapter": adapter_name,
            "fragment_budget_fallbacks": sum(
                r["budget_fallback"] for r in fragment_scores.values()
            ),
            "scope": "repeated2000 combined ranking"
            if limit == 2000
            else "fixed200 structural pilot; no publication eligibility",
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "proposal_queries_scored": len(pair_scores),
            "fingerprint_prior_ranking": fingerprint_prior_scores is not None,
            "sequence_ranking": sequence_scores is not None,
            "proposal_limit": proposal_limit,
            "fragment_limit": fragment_limit,
            "evidence_only": evidence_only,
            "critic_only": critic_only,
            "candidate_gate": candidate_gate,
            "relative_candidate_gate": relative_candidate_gate,
            "chemical_prior": chemical_prior,
            "fragment_weight": fragment_weight,
            "fragment_aggregation": "mean_normalized" if mean_fragments else "max_raw",
            "fragment_aggregation_source_sha256": execution_aggregation_sha256,
            "compare_current_first_fragment": compare_first,
            "fragment_gate_only": fragment_gate_only,
            "skip_impossible_critic_gate": skip_impossible,
            "evidence_gate": "Positive proposed first fragment score and distinct finite fragment scores required; missing/all tied scores never enable insertion"
            if evidence_only
            else None,
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
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--prefix", type=int, default=5)
    p.add_argument("--monomer", action="store_true")
    p.add_argument("--proposal-limit", type=int, choices=(100, 500), default=100)
    p.add_argument("--fragment-limit", type=int, default=100)
    p.add_argument("--evidence-only", action="store_true")
    p.add_argument("--critic-checkpoint", type=Path, default=CRITIC)
    p.add_argument("--dimer", action="store_true")
    p.add_argument("--candidate-gate", action="store_true")
    p.add_argument("--compare-first", action="store_true")
    p.add_argument("--fragment-gate-only", action="store_true")
    p.add_argument("--skip-impossible", action="store_true")
    p.add_argument("--relative-candidate-gate", action="store_true")
    p.add_argument("--pilot-selector", choices=("prefix", "hash"), default="prefix")
    p.add_argument("--critic-only", action="store_true")
    p.add_argument("--fingerprint-prior-scores", type=Path)
    p.add_argument("--sequence-ratio", type=float, choices=(0.0, 1.0), default=1.0)
    p.add_argument("--sequence-scores", type=Path)
    p.add_argument("--fragment-depth", type=int, choices=(2, 3), default=2)
    p.add_argument("--chemical-prior", action="store_true")
    p.add_argument("--mean-fragments", action="store_true")
    p.add_argument("--fragment-weight", type=float, choices=(0.5, 1.0), default=0.5)
    a = p.parse_args()
    print(
        json.dumps(
            run(
                a.output,
                a.incumbent,
                a.limit,
                a.prefix,
                a.monomer,
                a.proposal_limit,
                a.fragment_limit,
                a.evidence_only,
                a.critic_checkpoint,
                a.dimer,
                a.candidate_gate,
                a.compare_first,
                a.fragment_gate_only,
                a.skip_impossible,
                a.relative_candidate_gate,
                a.fragment_weight,
                a.mean_fragments,
                a.chemical_prior,
                a.fragment_depth,
                a.sequence_scores,
                a.sequence_ratio,
                a.fingerprint_prior_scores,
                a.critic_only,
                a.pilot_selector,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
