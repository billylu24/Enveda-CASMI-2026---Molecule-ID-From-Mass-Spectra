"""Unlabeled ChEMBL extension of the frozen0062 retrieval/generation ranking."""

import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from casmi_ml.chembl_fingerprint_prior import corrected_scores
from casmi_ml.chembl_fragment_pilot import informative_fragments, possible_critic_gate
from casmi_ml.chemistry import rerank
from casmi_ml.data import fingerprint, write_json
from casmi_ml.direct_models import DirectRanker, score_group
from casmi_ml.generation_inference import validated_confidence
from casmi_ml.generation_slots import insert_generated
from casmi_ml.inference import group_probability, validate_submission
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_monomer import MonomerMetFrag
from casmi_ml.ranking import CandidateIndex, neural_rank
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def select_high_fragment(current, tail, candidates, critic, fragments, fallback):
    if fallback:
        return tail, candidates[:3], "high_fragment_budget_keep_tail"
    proposed = rerank(
        candidates, {}, [], 0.5, top_n=len(candidates), fragment_scores=fragments
    )
    if not informative_fragments(proposed, fragments):
        return tail, candidates[:3], "high_fragment_evidence_keep_tail"
    if (
        fragments.get(proposed[0], 0) > fragments.get(current[0], 0)
        and critic[proposed[0]] > critic[current[0]] + 0.05
    ):
        return (
            insert_generated(current, proposed, 3, 3),
            proposed[:3],
            "high_fragment_inserted",
        )
    return current, [], "high_fragment_remove_tail"


@torch.inference_mode()
def extend(
    test_path,
    baseline_csv,
    full_path,
    routing_csv,
    output,
    root,
    config,
    deadline=None,
    fragment_deadline=None,
    java="java",
    cache=None,
    output_full_rankings=None,
):
    root, output = Path(root), Path(output)
    configure(threads=4)
    started = time.monotonic()
    test = pd.read_parquet(test_path)
    forbidden = {"inchikey14", "normalized_smiles", "molecular_formula", "fingerprint"}
    if forbidden & set(test.columns):
        raise ValueError("External inference requires unlabeled input")
    baseline = pd.read_csv(baseline_csv)
    validate_submission(test, baseline)
    raw = json.loads(Path(full_path).read_text())
    full = {r["molecule_id"]: r["smiles"] for r in raw}
    if len(full) != len(raw) or set(full) != set(test.molecule_id):
        raise ValueError("Current full ranking identifiers differ")
    indexed = baseline.set_index("molecule_id")
    for key, values in full.items():
        if values[:25] != indexed.loc[key, "smiles"].split(";"):
            raise ValueError("External handoff differs from frozen baseline CSV")
    routing = pd.read_csv(routing_csv)
    if routing.molecule_id.duplicated().any() or set(routing.molecule_id) != set(full):
        raise ValueError("Routing identifiers differ")
    confidence = validated_confidence(
        routing.set_index("molecule_id").confidence.to_dict()
    )
    paths = {
        name: root / config[name]
        for name in ("catalog", "encoder", "critic", "prior", "jar")
    }
    for name, path in paths.items():
        if digest(path) != config[name + "_sha256"]:
            raise ValueError(f"External {name} checksum changed")
    encoder, saved = load_deployment_checkpoint(paths["encoder"], "scale")
    encoder.eval()
    weights = torch.load(paths["critic"], map_location="cpu", weights_only=True)
    if (
        weights["encoder_sha256"] != config["encoder_sha256"]
        or weights["architecture"] != "fingerprint"
    ):
        raise ValueError("External critic/encoder binding differs")
    critic = DirectRanker("fingerprint").eval()
    critic.load_state_dict(weights["state_dict"])
    prior = np.load(paths["prior"])
    catalog = CandidateIndex(pd.read_parquet(paths["catalog"]))
    high_fragment = config.get("high_fragment")
    engine_class, engine_options = MonomerMetFrag, {}
    if high_fragment is not None:
        if high_fragment.get("prefix") != 3 or high_fragment.get("weight") != 0.5:
            raise ValueError("Only the frozen high fragment rule is supported")
        if high_fragment.get("backend") != "persistent":
            raise ValueError("High fragment requires the verified persistent backend")
        worker = root / high_fragment["worker_class"]
        if digest(worker) != high_fragment["worker_class_sha256"]:
            raise ValueError("Java worker bytecode changed")
        from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag

        engine_class = PersistentMonomerMetFrag
        engine_options = {"classes": worker.parent}
    fragmenter = engine_class(
        paths["jar"],
        cache or str(output) + ".external_fragment_cache",
        java=java,
        **engine_options,
    )
    rows, audit, final_full = [], [], []
    for key, group in test.groupby("molecule_id", sort=False):
        smiles = full[key]
        current = []
        lookup = {}
        for value in smiles:
            structure = Chem.MolFromSmiles(value)
            if structure is None:
                raise ValueError("Invalid current structure")
            candidate = Chem.MolToInchiKey(structure)[:14]
            if candidate in lookup:
                raise ValueError("Duplicate current full structure")
            current.append(candidate)
            lookup[candidate] = value
        status = "no_supported_external_candidates"
        inserted = []
        result = current
        if deadline is not None and time.monotonic() >= deadline:
            status = "total_budget_fallback"
        elif current:
            if len(catalog.cache) > 20000:
                catalog.cache.clear()
            query = group.drop(columns=["molecule_id"])
            pool, fps = candidate_window(catalog, query, "charge_aware_union")
            if len(pool):
                probability = group_probability(encoder, query, saved["preprocessing"])
                native = neural_rank(probability, pool.inchikey14.tolist(), fps)
                high = confidence[key] >= 0.5
                if high:
                    values = corrected_scores(probability, fps, prior)
                    native = [
                        pool.iloc[i].inchikey14
                        for i in sorted(
                            range(len(pool)),
                            key=lambda i: (-values[i], pool.iloc[i].inchikey14),
                        )
                    ]
                candidates = [k for k in native if k not in set(current)][:100]
                for candidate, structure in zip(
                    pool.inchikey14, pool.normalized_smiles
                ):
                    lookup.setdefault(candidate, structure)
                if candidates:
                    score_smiles = [lookup[current[0]]] + [
                        lookup[k] for k in candidates
                    ]
                    values = score_group(
                        critic,
                        encoder,
                        query,
                        saved["preprocessing"],
                        pd.DataFrame({"normalized_smiles": score_smiles}),
                        np.stack([fingerprint(s) for s in score_smiles]).astype(
                            np.float32
                        ),
                    )
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite external critic scores")
                    scores = dict(zip([current[0]] + candidates, values))
                    candidates = sorted(candidates, key=lambda k: (-scores[k], k))
                    status = "critic_gate_fallback"
                    if possible_critic_gate(candidates, scores, current[0], 0.05):
                        if high:
                            if scores[candidates[0]] > scores[current[0]] + 0.05:
                                inserted = candidates[:3]
                                result = insert_generated(current, candidates, 10, 3)
                                status = "high_tail_inserted"
                                if high_fragment is not None:
                                    fragments, fallback = score_group_merged(
                                        fragmenter,
                                        query.to_dict("records"),
                                        {
                                            k: lookup[k]
                                            for k in candidates + [current[0]]
                                        },
                                        deadline=fragment_deadline,
                                    )
                                    result, inserted, status = select_high_fragment(
                                        current,
                                        result,
                                        candidates,
                                        scores,
                                        fragments,
                                        fallback,
                                    )
                        else:
                            fragments, fallback = score_group_merged(
                                fragmenter,
                                query.to_dict("records"),
                                {k: lookup[k] for k in candidates + [current[0]]},
                                deadline=fragment_deadline,
                            )
                            proposed = rerank(
                                candidates,
                                {},
                                [],
                                0.5,
                                top_n=len(candidates),
                                fragment_scores=fragments,
                            )
                            status = (
                                "fragment_budget_fallback"
                                if fallback
                                else "fragment_evidence_fallback"
                            )
                            if (
                                not fallback
                                and informative_fragments(proposed, fragments)
                                and fragments.get(proposed[0], 0)
                                > fragments.get(current[0], 0)
                                and scores[proposed[0]] > scores[current[0]] + 0.05
                            ):
                                inserted = proposed[:3]
                                result = insert_generated(current, proposed, 3, 3)
                                status = "low_fragment_inserted"
        result_smiles = [lookup[k] for k in result]
        rows.append({"molecule_id": key, "smiles": ";".join(result_smiles[:25])})
        final_full.append({"molecule_id": key, "smiles": result_smiles})
        audit.append(
            {
                "molecule_id": key,
                "confidence": confidence[key],
                "status": status,
                "inserted": len(inserted),
            }
        )
    if high_fragment is not None:
        fragmenter.close()
    submission = pd.DataFrame(rows)
    validate_submission(test, submission)
    submission.to_csv(output, index=False)
    if output_full_rankings is not None:
        write_json(output_full_rankings, final_full)
    pd.DataFrame(audit).to_csv(str(output) + ".external.csv", index=False)
    write_json(
        str(output) + ".external.report.json",
        {
            "molecules": len(submission),
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "status_counts": pd.DataFrame(audit).status.value_counts().to_dict(),
            "source_sha256": digest(Path(__file__)),
            "config": config,
            "unlabeled_input": True,
            "independent_acceptance": False,
        },
    )
    return submission
