"""Complete existing catalog availability upper bound, separate from current/scored recall."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from casmi_ml.cached_structure_keys import structure_key
from casmi_ml.candidate_catalog import expanded_pool
from casmi_ml.data import write_json
from casmi_ml.full_gap_candidate_system import LOCAL_CONTROL
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.mass_candidates import mass_centers
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex, build_candidates
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import CATALOG, freeze
from casmi_ml.scale_experiment import COCONUT


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Local-only availability audit required")
    output = Path(output)
    config = json.loads((LOCAL_CONTROL / "external_config.json").read_text())
    sources = {
        "training_catalog": CATALOG,
        "coconut": COCONUT,
        "pubchemlite": Path("external/pubchemlite/structures.parquet"),
        "chembl": Path(config["catalog"]),
    }
    if digest(sources["chembl"]) != config["catalog_sha256"]:
        raise ValueError("Entire original external catalog required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "source_catalogs": {
                n: {"path": str(p), "sha256": digest(p)} for n, p in sources.items()
            },
            "composition_source_sha256": digest("casmi_ml/candidate_catalog.py"),
            "library_composition_sha256": digest("casmi_ml/ranking.py"),
            "window_source_sha256": digest("casmi_ml/mass_candidates.py"),
            "query_sha256": digest(CONTROL / "unknown/test.parquet"),
            "current_sha256": digest(CONTROL / "unknown/baseline_full.json"),
            "rankings_sha256": digest(LOCAL_CONTROL / "selected_rankings.json"),
            "rule": "Full original train-only library+entireCOCONUT+entirePubChemLite+entireChEMBL "
            "with original expanded_pool priority, no source download/expansion/query-specific subsets. "
            "Exact original charge-aware-union35ppm/.006Da mass centers from unlabeled query. "
            "NO fingerprint filtering: declared-key mass availability upper bound only. "
            "Truth enters aggregate only; retain2000 denominator including5 missing observable rows. "
            "Separate catalog membership, window upper bound, original current and final top25. "
            "No scoring/training/physical work/prediction changes or model selection.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    library = pd.read_parquet(CATALOG)
    frame = build_candidates(library, COCONUT, final=False)
    frame = expanded_pool(frame, pd.read_parquet(sources["pubchemlite"]))
    frame = expanded_pool(frame, pd.read_parquet(sources["chembl"]))
    index = CandidateIndex(frame)
    catalog_keys = set(frame.inchikey14)
    external_keys = set(
        pd.read_parquet(sources["chembl"], columns=["inchikey14"]).inchikey14
    )
    expected = json.loads((LOCAL_CONTROL / "selected_rankings.json").read_text())[
        "unknown"
    ]
    current = {
        r["molecule_id"]: [structure_key(s) for s in r["smiles"]]
        for r in json.loads((CONTROL / "unknown/baseline_full.json").read_text())
    }
    queries = pd.read_parquet(CONTROL / "unknown/test.parquet")
    groups = queries.groupby("molecule_id", sort=False)
    if len(expected) != 2000 or set(groups.groups) - set(expected):
        raise ValueError("Original complete2000 denominator required")
    missing = set(expected) - set(groups.groups)
    if any(expected[k] or k in current for k in missing):
        raise ValueError("Missing observable query must retain original empty ranking")
    counts, buckets = Counter(), Counter()
    for key in expected:
        if c.stopped():
            return None
        counts["molecules"] += 1
        counts["truth_in_complete_catalog"] += key in catalog_keys
        counts["truth_in_original_current"] += key in current.get(key, [])
        window = set()
        if key in groups.groups:
            query = groups.get_group(key).drop(columns="molecule_id")
            for center in mass_centers(query, "charge_aware_union"):
                window.update(index.query(center).inchikey14)
        else:
            counts["no_observable_query_rows"] += 1
        counts["truth_in_declared_mass_window_upper_bound"] += key in window
        counts["truth_in_current_or_declared_mass_window_upper_bound"] += (
            key in window or key in current.get(key, [])
        )
        if key not in current.get(key, []) and key not in external_keys:
            if key in window:
                buckets[
                    "absent_current_and_external_but_available_other_catalog_window"
                ] += 1
            elif key in catalog_keys:
                buckets[
                    "absent_current_and_external_in_complete_catalog_outside_window"
                ] += 1
            else:
                buckets["absent_current_and_complete_catalog"] += 1
        if key in window and key not in current.get(key, []):
            counts["declared_window_available_truth_absent_current"] += 1
            counts["window_absent_current_present_final"] += key in expected[key][:25]
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "complete_catalog_molecules": len(frame),
        "counts": dict(counts),
        "external_current_absence_breakdown": dict(buckets),
        "predictions_unchanged": True,
        "new_physical_requests": 0,
        "scope": "Declared identity/mass availability upper bound across full existing "
        "source catalogs, not valid fingerprints, neural shortlist or candidate-specific "
        "gate rejection. Missing query rows have no invented window. Repeated-development "
        "diagnosis only, no acceptance, accuracy gain or source/model activation.",
    }
    write_json(output / "report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
