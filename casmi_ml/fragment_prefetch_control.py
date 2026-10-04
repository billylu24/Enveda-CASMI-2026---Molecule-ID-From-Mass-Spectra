"""Complete existing scientific236 control for parallel fragment scheduling."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import casmi_ml.full_gap_candidate_system as base
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.portable_fragment_prefetch import extend_prefetched
from casmi_ml.research_protocol import freeze


def run(output, model):
    output = Path(output)
    freeze(
        output / "prefetch_sources.json",
        {
            "source_sha256": digest(__file__),
            "prefetch_source_sha256": digest("casmi_ml/portable_fragment_prefetch.py"),
            "base_system_source_sha256": digest(base.__file__),
            "rule": "Existing scientific236 original zero-score full3340 exact physical control; two independent original engines retain shared1200-second deadline and original request dictionaries. Cached complete control, not cold runtime proof. No model/science change.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch.object(base, "extend", new=extend_prefetched):
        report = base.run(output, model, control=True)
    if report:
        report["prefetch_source_sha256"] = digest(
            "casmi_ml/portable_fragment_prefetch.py"
        )
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
