"""Full3340 same-science control for streaming original physical request discovery."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml import full_gap_candidate_system as base
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze
from casmi_ml.streaming_fragment_prefetch import extend_streamed


def run(output, model):
    output = Path(output)
    freeze(
        output / "streaming_sources.json",
        {
            "source_sha256": digest(__file__),
            "streaming_sha256": digest("casmi_ml/streaming_fragment_prefetch.py"),
            "prefetch_sha256": digest("casmi_ml/portable_fragment_prefetch.py"),
            "rule": "Record and start unchanged physical requests during original critic/probability planning rather than after discovery. Same at most2 worker JVMs,2threads per worker, original60s timeout, shared1200s deadline, original candidate/request dictionaries and scientific gates. No prepackaged query evidence.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch.object(base, "extend", new=extend_streamed):
        result = base.run(output, model, control=True)
    if result is not None:
        result["streaming_source_sha256"] = digest(
            "casmi_ml/streaming_fragment_prefetch.py"
        )
        result["prefetch_source_sha256"] = digest(
            "casmi_ml/portable_fragment_prefetch.py"
        )
        write_json(output / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
