"""Complete3340 original236 identity control for balanced CPU/catalog runtime."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml import full_gap_candidate_system as base
from casmi_ml.balanced_cpu_fragment_runtime import extend_balanced
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze


def run(output, model):
    output = Path(output)
    freeze(
        output / "balanced_cpu_sources.json",
        {
            "source_sha256": digest(__file__),
            "runtime_source_sha256": digest(
                "casmi_ml/balanced_cpu_fragment_runtime.py"
            ),
            "streaming_source_sha256": digest(
                "casmi_ml/streaming_fragment_prefetch.py"
            ),
            "prefetch_source_sha256": digest("casmi_ml/portable_fragment_prefetch.py"),
            "rule": "Only external CPU torch scoring concurrency4->1 and same-invocation whole catalog/index reuse; no BLAS change. Original two physical JVMs each2candidate threads, original seeds/graphs/candidate order/deadlines. Cached3340 full ranks must exactly match original236 before any frozen75/cold400/platform400 runtime amendment.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch.object(base, "extend", new=extend_balanced):
        report = base.run(output, model, control=True)
    if report:
        report["balanced_cpu_source_sha256"] = digest(
            "casmi_ml/balanced_cpu_fragment_runtime.py"
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
