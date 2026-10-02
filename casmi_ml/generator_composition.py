"""Frozen decoder/formula-head composition with explicit conditioning checks."""

import argparse
import json
from pathlib import Path

import torch

from casmi_ml.data import write_json
from casmi_ml.generation_temperature_experiment import CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze


def compose(decoder_saved, formula_saved):
    for field in ["config", "vocabulary", "condition_dim"]:
        if decoder_saved[field] != formula_saved[field]:
            raise ValueError(f"Generator composition mismatch: {field}")
    a, b = decoder_saved["formula"], formula_saved["formula"]
    if a.keys() != b.keys():
        raise ValueError("Formula tensor names differ")
    for name, tensor in a.items():
        other = b[name]
        if (
            tensor.shape != other.shape
            or tensor.dtype != other.dtype
            or not torch.isfinite(other).all()
        ):
            raise ValueError("Formula tensor shape, dtype or finiteness mismatch")
    result = dict(decoder_saved)
    result["formula"] = {k: v.clone() for k, v in b.items()}
    return result


def run(output, formula_checkpoint):
    output, formula_checkpoint = Path(output), Path(formula_checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": 1,
        "source_sha256": digest(Path(__file__)),
        "decoder_checkpoint_sha256": digest(CHECKPOINT),
        "formula_checkpoint_sha256": digest(formula_checkpoint),
        "formula_checkpoint": str(formula_checkpoint),
        "holdout_used": False,
        "selection": "Fixed mass-consistency formula head composed with selected three-epoch decoder; no new training or coefficient sweep",
    }
    freeze(output / "protocol.json", protocol)
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    decoder_saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    formula_saved = torch.load(
        formula_checkpoint, map_location="cpu", weights_only=True
    )
    saved = compose(decoder_saved, formula_saved)
    saved["generator_composition"] = protocol
    saved["initial_report"] = saved.pop("report", None)
    saved["report"] = {
        "scope": "Fixed head composition; performance requires structural evaluation"
    }
    torch.save(saved, output / "model.pt")
    result = {
        "checkpoint_sha256": digest(output / "model.pt"),
        "decoder_all_tensors_unchanged": True,
        "formula_changed_tensors": sum(
            not torch.equal(v, saved["formula"][k])
            for k, v in decoder_saved["formula"].items()
        ),
        "gpu_training_seconds": 0,
        "diagnostic_only": True,
        "independent_acceptance": False,
        "performance_acceptance": "Compare structural pilot against current selected decoder, then full2000 paired gate",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formula-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.formula_checkpoint), indent=2))


if __name__ == "__main__":
    main()
