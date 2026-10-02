"""Central molecule-level accounting for fresh and repeated research cohorts."""

import fcntl
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest


def record_usage(registry, cohort, split, round_id):
    registry, cohort = Path(registry), Path(cohort)
    registry.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cohort / "cohorts.json").read_text())[split]
    if digest(cohort / f"{split}.parquet") != manifest["sha256"]:
        raise ValueError("Registered cohort checksum changed")
    keys = sorted(set(manifest["keys"]))
    if len(keys) != manifest["molecules"]:
        raise ValueError("Cohort keys/count inconsistent")
    with registry.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = (
            json.loads(registry.read_text())
            if registry.exists()
            else {"used_keys": [], "uses": []}
        )
        previous = next(
            (
                r
                for r in state["uses"]
                if r["round"] == round_id and r["split"] == split
            ),
            None,
        )
        if previous:
            if previous["sha256"] != manifest["sha256"]:
                raise ValueError("Round cohort changed")
            return previous
        overlap = len(set(keys) & set(state["used_keys"]))
        result = {
            "round": round_id,
            "split": split,
            "sha256": manifest["sha256"],
            "molecules": len(keys),
            "previously_used_molecules": overlap,
            "fresh_molecules": len(keys) - overlap,
            "label": "repeated_development"
            if overlap
            else "first_use_in_registry_only; historical manifests must also be audited",
        }
        state["used_keys"] = sorted(set(keys) | set(state["used_keys"]))
        state["uses"].append(result)
        write_json(registry, state)
        return result
