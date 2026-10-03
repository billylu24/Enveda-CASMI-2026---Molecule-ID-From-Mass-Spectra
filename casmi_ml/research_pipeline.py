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
    critic = generation.get("critic")
    critic_path = recipe_path.parent / critic["checkpoint"] if critic else None
    if critic and digest(critic_path) != critic["sha256"]:
        raise ValueError("Generated critic checkpoint changed")
    started = time.monotonic()
    baseline = output.with_name("retrieval.csv")
    full = output.with_name("retrieval_full.json")
    retrieve_predict(recipe_path, data_dir, coconut, baseline, full_rankings=full)
    test = locate(Path(data_dir) / "test.parquet", "test.parquet")
    remaining = max(0.001, generation["total_seconds"] - (time.monotonic() - started))
    external = recipe.get("external_routed")
    generated_output = output.with_name("generated_base.csv") if external else output
    generated_full = output.with_name("generated_base_full.json") if external else None
    result = generate_predict(
        generation_path,
        test,
        baseline,
        generated_output,
        samples=generation["samples"],
        seconds=remaining,
        encoder_path=recipe_path.parent / recipe["checkpoint"],
        prefix=generation["prefix"],
        slots=generation["slots"],
        full_rankings=full,
        open_protected=generation.get("open_protected", False),
        frequency_weight=generation.get("frequency_weight", 0.0),
        token_length_exponent=generation.get("token_length_exponent", 0.0),
        critic_checkpoint=critic_path,
        critic_weight=critic["weight"] if critic else 0.0,
        adaptive_prefix=generation.get("adaptive_prefix"),
        expanded_prefix=generation.get("expanded_prefix"),
        first_gate=generation.get("first_gate"),
        output_full_rankings=generated_full,
    )
    report = json.loads(Path(str(generated_output) + ".report.json").read_text())
    if external:
        from casmi_ml.chembl_routed_inference import extend

        result = extend(
            test,
            generated_output,
            generated_full,
            str(baseline) + ".routing.csv",
            output,
            recipe_path.parent,
            external,
            deadline=started + generation["total_seconds"],
            fragment_deadline=started + recipe["chemistry"]["fragment_seconds"],
        )
        report["external_routed"] = json.loads(
            Path(str(output) + ".external.report.json").read_text()
        )
        generated_full.unlink()
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
