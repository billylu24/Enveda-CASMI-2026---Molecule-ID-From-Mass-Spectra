"""Experimental worker: original seeded generator with one Torch CPU thread."""

import argparse
from unittest.mock import patch

from casmi_ml import generation_inference as original
from casmi_ml.generation_handoff_stream import run_worker as original_worker


def run_worker(config):
    configure = original.configure

    def bounded(seed=42, threads=4):
        return configure(seed=seed, threads=1)

    # Original inspected AST, seeds, models, CUDA work, sample and deadline stay
    # unchanged. CPU reduction differences require complete output controls.
    with patch.object(original, "configure", new=bounded):
        return original_worker(config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", required=True)
    run_worker(p.parse_args().worker)


if __name__ == "__main__":
    main()
