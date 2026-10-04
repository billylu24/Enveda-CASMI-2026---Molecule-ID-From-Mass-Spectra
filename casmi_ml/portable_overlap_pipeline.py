"""Local original236 experiment: overlap generation with unchanged physical discovery."""

import argparse
import inspect
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.generation_fragment_overlap import generate_with_physical_planning
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.portable_reference_pipeline import run as original_run


def run(recipe, data_dir, coconut, output, force_partitions=False):
    recipe, output = Path(recipe), Path(output)
    config = json.loads(recipe.read_text())
    if not config["generation"] or config["chemistry"]["fragment_seconds"] != 1200:
        raise ValueError(
            "Original frozen generation and1200 physical deadline required"
        )
    state = {}

    def generate(*args, **kwargs):
        caller = inspect.currentframe().f_back
        if (
            caller.f_code.co_name != "predict"
            or caller.f_globals["__name__"] != "casmi_ml.research_pipeline"
        ):
            raise ValueError("Original pipeline's shared start time required")
        started = caller.f_locals["started"]
        value, pool, stack, details = generate_with_physical_planning(
            args,
            kwargs,
            root=recipe.parent,
            config=config["external_routed"],
            deadline=started + config["chemistry"]["fragment_seconds"],
            total_deadline=started + config["generation"]["total_seconds"],
            final_output=output,
            force_partition=force_partitions,
        )
        state.update(pool=pool, stack=stack, details=details)
        return value

    def extend(*args, **kwargs):
        from casmi_ml import portable_fragment_prefetch as physical

        if (
            "pool" not in state
            or kwargs.get("fragment_deadline") != state["pool"].deadline
        ):
            raise ValueError("Unchanged original physical deadline required")

        class Consumer(PersistentMonomerMetFrag):
            def _run(self, *request):
                return state["pool"].fetch(*request)

        with patch(
            "casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=Consumer
        ):
            return physical.serial_extend(*args, **kwargs)

    try:
        with (
            patch("casmi_ml.research_pipeline.generate_predict", new=generate),
            patch("casmi_ml.portable_streaming_pipeline.extend_streamed", new=extend),
        ):
            result = original_run(recipe, data_dir, coconut, output)
    finally:
        if "pool" in state:
            state["pool"].close()
        if "stack" in state:
            state["stack"].close()
    report_path = Path(str(output) + ".report.json")
    report = json.loads(report_path.read_text())
    report["generation_physical_overlap"] = {
        **state["details"],
        "stream_source_sha256": digest("casmi_ml/generation_handoff_stream.py"),
        "overlap_source_sha256": digest("casmi_ml/generation_fragment_overlap.py"),
        "pipeline_source_sha256": digest(__file__),
        "scope": "Original per-molecule sampling and complete handoffs, same physical request dictionaries/models/recipe/heap/twoJVM/two candidate threads/shared deadlines. Runtime-only in-invocation atomic events, no query evidence packaged. Experimental local-only; require full3340/fresh75/cold400/physical dictionary/source/resource proof before any future release amendment.",
    }
    write_json(report_path, report)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--force-partitions", action="store_true")
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
