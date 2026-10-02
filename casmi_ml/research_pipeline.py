"""Offline retrieval plus frozen generated slots, with a shared total deadline."""

import argparse
import json
import resource
import time
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.generation_inference import predict as generate_predict
from casmi_ml.inference import locate
from casmi_ml.metfrag import digest
from casmi_ml.secondary_inference import predict as retrieve_predict


def predict(recipe_path, data_dir, coconut, output):
    recipe_path, output = Path(recipe_path), Path(output)
    recipe = json.loads(recipe_path.read_text())
    generation = recipe.get("generation")
    if generation is None:
        return retrieve_predict(recipe_path, data_dir, coconut, output)
    if generation["sampling"] != "spectrum_hash_v1_and_shared_group_forward_v2":
        raise ValueError("Generation evaluation/inference sampling protocol mismatch")
    generation_path = recipe_path.parent / generation["checkpoint"]
    if digest(generation_path) != generation["sha256"]:
        raise ValueError("Generation weights changed")
    started = time.monotonic()
    baseline = output.with_name("retrieval.csv")
    full = output.with_name("retrieval_full.json")
    retrieve_predict(recipe_path, data_dir, coconut, baseline, full_rankings=full)
    test = locate(Path(data_dir) / "test.parquet", "test.parquet")
    remaining = max(0.001, generation["total_seconds"] - (time.monotonic() - started))
    result = generate_predict(
        generation_path,
        test,
        baseline,
        output,
        samples=generation["samples"],
        seconds=remaining,
        encoder_path=recipe_path.parent / recipe["checkpoint"],
        prefix=generation["prefix"],
        slots=generation["slots"],
        full_rankings=full,
        open_protected=generation.get("open_protected", False),
        frequency_weight=generation.get("frequency_weight", 0.0),
    )
    report = json.loads(Path(str(output) + ".report.json").read_text())
    report.update(
        seconds=time.monotonic() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        generation=generation,
        independent_acceptance=False,
    )
    write_json(str(output) + ".report.json", report)
    # Full local ranking cache is only an internal handoff, never a public output asset.
    full.unlink()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--coconut", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    predict(a.recipe, a.data_dir, a.coconut, a.output)


if __name__ == "__main__":
    main()
