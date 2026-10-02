"""Frozen decoder weight interpolation; formula and conditioner remain unchanged."""

import argparse
import json
from pathlib import Path

import torch

from casmi_ml.data import write_json
from casmi_ml.generation_temperature_experiment import CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, freeze


def interpolate_decoder(original, selected, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Decoder interpolation fraction must be between zero and one")
    if original.keys() != selected.keys():
        raise ValueError("Decoder tensor names differ")
    result = {}
    for name, value in original.items():
        other = selected[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("Decoder tensor shape or dtype mismatch")
        if not value.is_floating_point():
            if not torch.equal(value, other):
                raise ValueError("Cannot interpolate differing nonfloating tensors")
            result[name] = value.clone()
        elif not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError("Cannot interpolate nonfinite tensors")
        else:
            result[name] = torch.lerp(value, other, alpha)
    return result


def run(output, alpha=0.5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_path = ROOT / "generation/smiles_42/model.pt"
    protocol = {
        "version": 1,
        "source_sha256": digest(Path(__file__)),
        "alpha": alpha,
        "original_sha256": digest(original_path),
        "selected_sha256": digest(CHECKPOINT),
        "target": "Decoder tensors only; formula predictor, encoder configuration and vocabulary unchanged",
        "trained_on_development": False,
        "holdout_used": False,
    }
    freeze(output / "protocol.json", protocol)
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    original = torch.load(original_path, map_location="cpu", weights_only=True)
    selected = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if (
        original["config"] != selected["config"]
        or original["vocabulary"] != selected["vocabulary"]
        or original["condition_dim"] != selected["condition_dim"]
    ):
        raise ValueError("Generator conditioning or vocabulary differs")
    if original["formula"].keys() != selected["formula"].keys() or any(
        not torch.equal(v, selected["formula"][k])
        for k, v in original["formula"].items()
    ):
        raise ValueError("Interpolation requires identical formula predictor")
    original["decoder"] = interpolate_decoder(
        original["decoder"], selected["decoder"], alpha
    )
    original["decoder_interpolation"] = protocol
    original["initial_report"] = original.pop("report", None)
    original["report"] = {
        "scope": "Untrained fixed decoder interpolation; performance requires structural evaluation"
    }
    torch.save(original, output / "model.pt")
    result = {
        "alpha": alpha,
        "checkpoint_sha256": digest(output / "model.pt"),
        "formula_all_tensors_unchanged": True,
        "vocabulary_unchanged": True,
        "gpu_training_seconds": 0,
        "diagnostic_only": True,
        "independent_acceptance": False,
        "performance_acceptance": "Requires paired sampling and structural evaluation against current best",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.alpha), indent=2))


if __name__ == "__main__":
    main()
