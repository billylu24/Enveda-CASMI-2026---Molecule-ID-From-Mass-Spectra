"""Finalize a complete local candidate-score cycle and activate only its real winner."""

import argparse
import json

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.intensity_candidate_activation import activate
from casmi_ml.local_research_snapshot import refresh as snapshot
from casmi_ml.query_cycle_aggregate import refresh as aggregate
from casmi_ml.research_loop import Controller


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chemical", default="0316_intensity_candidate_chemical_system")
    p.add_argument("--mask", default="0317_intensity_candidate_mask_system")
    a = p.parse_args()
    c = Controller()
    if not c.local_only():
        raise ValueError("Local completion is local-only")
    directory = wait_for_complete_system(c, a.chemical)
    if directory is None:
        return
    result = json.loads((directory / "decision.json").read_text())
    activate(c, directory, result)
    if wait_for_complete_system(c, a.mask) is None:
        return
    snapshot()
    aggregate()
    print(
        "Finalized complete local candidate cycle",
        a.chemical,
        "winner",
        bool(result.get("winner")),
        flush=True,
    )


if __name__ == "__main__":
    main()
