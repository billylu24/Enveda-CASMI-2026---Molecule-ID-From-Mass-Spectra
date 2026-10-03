"""Run the currently verified local external rule on a label-free baseline handoff."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from casmi_ml.chembl_routed_inference import extend
from casmi_ml.data import write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_system_evaluation import resolve_local_incremental
from casmi_ml.metfrag import digest
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_loop import Controller


def verified_config(c):
    if not c.local_only():
        raise ValueError("Verified local extension requires local-only mode")
    proof = c.read().get("local_verified_inference", {})
    if not proof or not proof.get("local_only"):
        raise ValueError("No fully replayed local inference rule is active")
    directory = c.root / "rounds" / proof["evidence_round"]
    for name, field in [
        ("report.json", "replay_report_sha256"),
        ("selected_rankings.json", "rankings_sha256"),
        ("external_config.json", "external_config_sha256"),
    ]:
        if digest(directory / name) != proof[field]:
            raise ValueError("Local inference evidence changed: " + name)
    for name, field in [
        ("casmi_ml/chembl_routed_inference.py", "inference_sha256"),
        ("casmi_ml/low_leader_inference.py", "low_leader_source_sha256"),
    ]:
        if digest(name) != proof[field]:
            raise ValueError("Local inference source changed; replay again: " + name)
    if (
        proof.get("array_source_sha256")
        and digest("casmi_ml/candidate_array_order.py") != proof["array_source_sha256"]
    ):
        raise ValueError("Local array ordering source changed; replay again")
    if (
        proof.get("identity_source_sha256")
        and digest("casmi_ml/cached_structure_keys.py")
        != proof["identity_source_sha256"]
    ):
        raise ValueError("Local identity cache source changed; replay again")
    if c.read()["development_best"]["round"] != proof["scientific_round"]:
        raise ValueError("Active verified inference no longer matches current best")
    return json.loads((directory / "external_config.json").read_text()), proof


def run(test, baseline, full, routing, reference, output):
    c = Controller()
    config, proof = verified_config(c)
    benchmark, inherited, _ = resolve_local_incremental("verified-local", None)
    engines = []

    def engine(*args, **kwargs):
        obj = DevelopmentIncrementalMetFrag(*args, benchmark=benchmark, **kwargs)
        engines.append(obj)
        return obj

    started = time.monotonic()
    with ExitStack() as stack:
        from casmi_ml.blas_runtime import resolve as resolve_blas

        stack.enter_context(resolve_blas(c))
        from casmi_ml.validation_cache_runtime import resolve as resolve_validation

        cached_validator = resolve_validation(c)
        if cached_validator is not None:
            stack.enter_context(
                patch(
                    "casmi_ml.chembl_routed_inference.validate_submission",
                    new=cached_validator,
                )
            )
        stack.enter_context(
            patch("casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=engine)
        )
        from casmi_ml.packed_candidate_runtime import resolve as resolve_candidates

        candidate_constructor = resolve_candidates(c, config["catalog_sha256"])
        if candidate_constructor is not None:
            stack.enter_context(
                patch(
                    "casmi_ml.chembl_routed_inference.CandidateIndex",
                    new=candidate_constructor,
                )
            )
        from casmi_ml.candidate_compatibility_activation import resolve_score

        compatibility_score = resolve_score(config, proof)
        if compatibility_score is not None:
            stack.enter_context(
                patch(
                    "casmi_ml.chembl_routed_inference.score_group",
                    new=compatibility_score,
                )
            )
        from casmi_ml.extended_candidate_activation import (
            resolve_score as resolve_extended_score,
        )

        extended_score = resolve_extended_score(config, proof)
        if compatibility_score is not None and extended_score is not None:
            raise ValueError(
                "Only one fully evaluated candidate chemistry bank may be active"
            )
        if extended_score is not None:
            stack.enter_context(
                patch(
                    "casmi_ml.chembl_routed_inference.score_group", new=extended_score
                )
            )
        from casmi_ml.intensity_candidate_activation import (
            resolve_score as resolve_intensity_score,
        )

        intensity_score = resolve_intensity_score(config, proof)
        if intensity_score is not None:
            if compatibility_score is not None or extended_score is not None:
                raise ValueError(
                    "Only one verified candidate chemistry bank may be active"
                )
            stack.enter_context(
                patch(
                    "casmi_ml.chembl_routed_inference.score_group", new=intensity_score
                )
            )
        query_spec = config.get("query_residual")
        if query_spec:
            from casmi_ml.diagnostic_query_ranker import (
                load_query_ranker,
                score_chemical_group,
            )

            if (
                digest(query_spec["original_critic"])
                != query_spec["original_critic_sha256"]
            ):
                raise ValueError("Query residual original critic source changed")
            for name, field in [
                ("casmi_ml/diagnostic_query_ranker.py", "ranker_source_sha256"),
                ("casmi_ml/diagnostic_query_residual.py", "residual_source_sha256"),
            ]:
                if digest(name) != query_spec[field]:
                    raise ValueError(
                        "Query residual implementation changed; replay again"
                    )
            if (
                query_spec.get("strength_transform_source_sha256")
                and digest("casmi_ml/diagnostic_strength_transform.py")
                != query_spec["strength_transform_source_sha256"]
            ):
                raise ValueError(
                    "Query strength transform source changed; replay again"
                )
            ranker, _ = load_query_ranker(
                config["critic"], query_spec["original_critic"]
            )
            stack.enter_context(
                patch(
                    "casmi_ml.chembl_routed_inference.DirectRanker",
                    new=lambda architecture: ranker,
                )
            )
            stack.enter_context(
                patch(
                    "casmi_ml.chembl_routed_inference.score_group",
                    new=score_chemical_group,
                )
            )
        result = extend(
            test,
            baseline,
            full,
            routing,
            output,
            Path("."),
            config,
            deadline=started + 3600,
            fragment_deadline=started + 1200,
            java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
            cache="artifacts/chemistry_20261001/chembl_metfrag_cache",
            output_full_rankings=str(output) + ".full.json",
            reference=ReferenceIndex(reference),
        )
    failures = set().union(*(e.failed_physical_requests for e in engines))
    report = json.loads(Path(str(output) + ".external.report.json").read_text())
    budgets = sum(
        n for status, n in report["status_counts"].items() if "budget" in status
    )
    write_json(
        str(output) + ".local_verified.json",
        {
            "local_only": True,
            "scientific_round": proof["scientific_round"],
            "evidence_round": proof["evidence_round"],
            "development_budget_fallbacks": budgets,
            "new_failure_contexts": len(failures - inherited),
            "complete_evidence": not budgets and not (failures - inherited),
            "scope": "Verified local external extension of supplied baseline; new input outputs do not establish accuracy or cold full-pipeline resource acceptance",
            "remote_publication": False,
            "independent_acceptance": False,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["test", "baseline", "full", "routing", "reference", "output"]:
        parser.add_argument("--" + name, required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
