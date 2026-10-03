"""Independently attribute historical control drift without accepting model gains."""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_critic_slots import score_cache_key
from casmi_ml.chembl_high_fragment_pilot import HIGH, fragment_structures
from casmi_ml.chembl_high_strong_slots import CLASSES, JAR, JAVA
from casmi_ml.chembl_routed_inference import select_high_fragment
from casmi_ml.chemistry import rerank
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.diagnostic_system_evaluation import BASE, BEST
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_protocol import ROOT, freeze
from casmi_ml.strong_slot_inference import reference_supported_keys, select_strong_slots

FRAGMENT = Path("artifacts/research_loop/rounds/0163_high_fragment_completion")


@lru_cache(maxsize=200000)
def structure_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid source structure")
    return Chem.MolToInchiKey(mol)[:14]


def normalize_ranking(order, structures):
    result = [structure_key(structures[k]) for k in order]
    if len(set(result)) != len(result):
        raise ValueError("Source identity normalization produces duplicates")
    return result


def run(output, observed):
    output, observed = Path(output), Path(observed)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "incumbent_rankings_sha256": digest(BEST / "selected_rankings.json"),
            "observed_rankings_sha256": digest(observed / "selected_rankings.json"),
            "fragment_cache_sha256": digest(FRAGMENT / "fragment_scores.json"),
            "rule": "Normalize all historical structure identities from actual original SMILES; independently recompute all active frozen empty nonbudget high-fragment groups with same physical inputs and unchanged181 rules. Reject any other control drift. No query truth consulted.",
            "local_only": True,
            "diagnostic_only": True,
            "independent_acceptance": False,
        },
    )
    original = json.loads((BASE / "low_rankings.json").read_text())["baseline"]
    routed = json.loads((BASE / "selected_rankings.json").read_text())
    incumbent = json.loads((BEST / "selected_rankings.json").read_text())
    actual = json.loads((observed / "selected_rankings.json").read_text())
    config = json.loads((HIGH / "protocol.json").read_text())
    proposals = json.loads(Path(config["proposal_path"]).read_text())
    pairs = json.loads((HIGH / "critic_scores.json").read_text())
    fragments = json.loads((FRAGMENT / "fragment_scores.json").read_text())
    dependencies = json.loads((FRAGMENT / "protocol.json").read_text())[
        "dependencies_sha256"
    ]
    binding = {k: config[k] for k in ("encoder_sha256", "critic_sha256")}
    generated = {
        r["key"]: {c["key"]: c["smiles"] for c in r["candidates"]}
        for r in json.loads(GENERATED.read_text())
    }
    needed = {
        k for mode in incumbent.values() for order in mode.values() for k in order
    }
    for mode in original.values():
        for key, order in mode.items():
            needed.update([k for k in proposals[key] if k not in set(order)][:100])
    external_frame = pd.read_parquet(
        DERIVED, columns=["inchikey14", "normalized_smiles"]
    )
    external = (
        external_frame[external_frame.inchikey14.isin(needed)]
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    del external_frame
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    selected, audits, summary = {}, {}, {}
    engine = PersistentMonomerMetFrag(
        JAR,
        "artifacts/chemistry_20261001/chembl_metfrag_cache",
        java=str(JAVA),
        classes=CLASSES,
        threads=2,
    )
    deadline = time.monotonic() + 1200
    try:
        for mode in ("unknown", "known"):
            lookup = candidate_lookup(ROOT, "researchdev", mode)
            for r in pd.read_parquet(
                "external/pubchemlite/structures.parquet",
                columns=["inchikey14", "normalized_smiles"],
            ).itertuples():
                lookup.setdefault(r.inchikey14, r.normalized_smiles)
            confidence = {
                r["key"]: r["confidence"]
                for r in json.loads(
                    (ROOT / f"researchdev_{mode}_chemical.json").read_text()
                )
            }
            reference = ReferenceIndex(
                "artifacts/research_loop/rounds/0001_mass_v2/reference/" + mode
            )
            selected[mode], audits[mode] = {}, {}
            for key, order in incumbent[mode].items():
                original_keys = set(original[mode][key])
                structures = {
                    k: (
                        generated[key].get(k, lookup.get(k))
                        if k in original_keys
                        else external.get(k)
                    )
                    for k in order
                }
                if any(v is None for v in structures.values()):
                    raise ValueError("Historical selected structure unavailable")
                normalized = normalize_ranking(order, structures)
                detail = {
                    "identity_changes": sum(a != b for a, b in zip(order, normalized))
                }
                prior = original[mode][key]
                if prior and confidence[key] >= 0.5 and routed[mode][key] != prior:
                    native = [k for k in proposals[key] if k not in set(prior)][:100]
                    values = pairs[score_cache_key(key, prior[0], native, binding)]
                    candidates = sorted(native, key=lambda k: (-values[k], k))
                    physical = fragment_structures(
                        key, prior[0], candidates, external, lookup, generated
                    )
                    cache_key = score_cache_key(
                        "high_relative_merged_actual_first_v2:" + key,
                        prior[0],
                        candidates,
                        {"structures": physical, "dependencies": dependencies},
                    )
                    item = fragments[cache_key]
                    if not item["scores"] and not item["budget_fallback"]:
                        query = frame.iloc[groups[key]].drop(
                            columns=[
                                c
                                for c in (
                                    "inchikey14",
                                    "normalized_smiles",
                                    "fingerprint",
                                    "molecular_formula",
                                )
                                if c in frame
                            ]
                        )
                        scores, fallback = score_group_merged(
                            engine,
                            query.to_dict("records"),
                            physical,
                            deadline=deadline,
                        )
                        if fallback:
                            raise ValueError(
                                "Completion audit exhausted fragment budget"
                            )
                        result, _, status = select_high_fragment(
                            prior, routed[mode][key], candidates, values, scores, False
                        )
                        if status == "high_fragment_inserted":
                            original_structures = {
                                k: generated[key].get(k, lookup.get(k))
                                for k in prior[:3]
                            }
                            originals, fallback = score_group_merged(
                                engine,
                                query.to_dict("records"),
                                original_structures,
                                deadline=deadline,
                            )
                            proposed = rerank(
                                candidates,
                                {},
                                [],
                                0.5,
                                top_n=len(candidates),
                                fragment_scores=scores,
                            )
                            result, _, status = select_strong_slots(
                                prior,
                                result,
                                proposed,
                                values,
                                scores,
                                originals,
                                fallback,
                                reference_supported_keys(reference, query),
                            )
                        structures.update(physical)
                        normalized = normalize_ranking(result, structures)
                        detail.update(
                            frozen_empty_evidence=True,
                            completed_score_count=len(scores),
                            independently_selected_status=status,
                        )
                selected[mode][key] = normalized
                if detail["identity_changes"] or detail.get("frozen_empty_evidence"):
                    audits[mode][key] = detail
            if set(selected[mode]) != set(actual[mode]):
                raise ValueError("Observed control cohort differs")
            unexplained = [
                k for k in selected[mode] if selected[mode][k] != actual[mode][k]
            ]
            summary[mode] = {
                "molecules": len(selected[mode]),
                "identity_affected_groups": sum(
                    bool(v["identity_changes"]) for v in audits[mode].values()
                ),
                "active_empty_evidence_groups": sum(
                    v.get("frozen_empty_evidence", False) for v in audits[mode].values()
                ),
                "unexplained_differences": len(unexplained),
                "full_control_exact": not unexplained,
            }
            write_json(output / f"private_{mode}_unexplained.json", unexplained)
            print(mode, summary[mode], flush=True)
    finally:
        engine.close()
    write_json(output / "private_attribution.json", audits)
    write_json(output / "selected_rankings.json", selected)
    write_json(
        output / "report.json",
        {
            "summary": summary,
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "control_passed": all(v["full_control_exact"] for v in summary.values()),
        },
    )
    if not all(v["full_control_exact"] for v in summary.values()):
        raise ValueError(
            "Unexplained historical control drift remains; residual arms prohibited"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--observed", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
