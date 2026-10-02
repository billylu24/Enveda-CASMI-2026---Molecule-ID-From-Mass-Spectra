"""Replay expansion rank code on actual low-confidence development queries."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.candidate_catalog import expanded_pool
from casmi_ml.chemistry_experiment import reference_records
from casmi_ml.coverage_experiment import EXTERNAL
from casmi_ml.coverage_inference import expanded_chemical_rank, merge_expanded
from casmi_ml.data import write_json
from casmi_ml.inference import group_probability
from casmi_ml.metfrag import MetFrag
from casmi_ml.ranking import CandidateIndex, ReferenceIndex, build_candidates
from casmi_ml.research_protocol import CATALOG, ENCODER, ROOT
from casmi_ml.scale_experiment import COCONUT
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def run(directory, limit=25):
    directory = Path(directory)
    configure(threads=4)
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
    protocol = json.loads((directory / "protocol.json").read_text())
    incumbent = Path(protocol["incumbent_directory"])
    catalog = pd.read_parquet(CATALOG)
    external = pd.read_parquet(EXTERNAL)
    index = CandidateIndex(
        expanded_pool(build_candidates(catalog, COCONUT, final=False), external)
    )
    structures = index.catalog.set_index("inchikey14").normalized_smiles.to_dict()
    reference = ReferenceIndex(incumbent / "reference/unknown")
    fragmenter = MetFrag(
        "external/metfrag/MetFragCommandLine-2.6.11.jar", ROOT / "metfrag_cache"
    )
    controls = {r["key"]: r for r in reference_records(ROOT, "researchdev", "unknown")}
    rows = json.loads((directory / "unknown_records.json").read_text())
    chosen = [r for r in rows if controls[r["key"]]["confidence"] < 0.5][:limit]
    matches = 0
    for row in chosen:
        key = row["key"]
        group = frame.iloc[groups[key]]
        probability = group_probability(model, group, checkpoint["preprocessing"])
        rank, _, fallback = expanded_chemical_rank(
            group,
            controls[key]["rankings"]["coconut15"],
            probability,
            index,
            reference,
            structures,
            fragmenter,
        )
        base = row["variants"]["baseline"]["ranking"]
        valid = all(
            merge_expanded(base, rank, weight, fallback)
            == row["variants"][f"expansion_{weight:g}"]["ranking"]
            for weight in [0.25, 0.5, 1.0]
        )
        matches += valid
    result = {
        "valid": matches == len(chosen),
        "molecules": len(chosen),
        "complete_ranking_matches": matches,
        "scope": "Actual CPU/MetFrag shared deployment code replay; development implementation consistency only",
    }
    write_json(directory / "replay.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--limit", type=int, default=25)
    a = p.parse_args()
    print(run(a.directory, a.limit))


if __name__ == "__main__":
    main()
