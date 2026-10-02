"""Sequential local research workflow; never uploads or submits to Kaggle."""

import argparse
import json
import time
from pathlib import Path

import torch

from casmi_ml.chemistry_experiment import accept, develop
from casmi_ml.data import write_json
from casmi_ml.generation_experiment import generate
from casmi_ml.generation_experiment import train as train_generation
from casmi_ml.metfrag import MetFrag, digest
from casmi_ml.representation_experiment import train
from casmi_ml.research_evaluation import (
    generated_report,
    representation_accept,
    select_representations,
)
from casmi_ml.research_protocol import ROOT, freeze, prepare


def run(root=ROOT, jar=None, generation_limit=100, generation_samples=128):
    root = Path(root)
    prepare(root)
    spec = {
        "version": 2,
        "generation_dev_limit": generation_limit,
        "generation_samples": generation_samples,
        "representation_methods": ["supervised", "masked", "dino"],
        "pretrain_epochs": 20,
        "finetune_epochs": 30,
        "generation_epochs": 30,
        "stage_seconds": 86400,
        "seed": 42,
        "repeat_seeds": [43, 44],
        "jar_sha256": digest(jar) if jar else None,
        "no_automatic_submission": True,
    }
    freeze(root / "workflow.json", spec)
    start = time.monotonic()
    fragmenter = MetFrag(jar, root / "metfrag_cache") if jar else None
    develop(root, fragmenter)
    for method in spec["representation_methods"]:
        train(root, method=method)
    selected = select_representations(root)
    if selected["winner"]:
        for seed in spec["repeat_seeds"]:
            train(root, method=selected["winner"]["method"], seed=seed)
    generation = train_generation(root)
    # Freeze every development choice before evaluating any holdout labels.
    joint = {
        "chemical_selection_sha256": digest(root / "chemical_selection.json"),
        "representation_selection_sha256": digest(
            root / "representation_selection.json"
        ),
        "generation_checkpoint_sha256": digest(root / "generation/smiles_42/model.pt"),
        "generation_selection_metric": "development token NLL; exact retrieval performance reported separately",
        "generation_eval_limit": generation_limit,
        "generation_samples": generation_samples,
    }
    freeze(root / "joint_selection.json", joint)
    generated = generate(
        root, limit=generation_limit, samples=generation_samples, fragmenter=fragmenter
    )
    gen_report = generated_report(root, generated)
    write_json(root / "generation/development_report.json", gen_report)
    # A generation pilot does not open final holdout unless merged development MRR improves.
    if (
        gen_report["merged"]["mrr25"] > gen_report["baseline"]["mrr25"]
        and gen_report["known"] is not None
        and gen_report["known"]["merged"]["mrr25"]
        >= gen_report["known"]["baseline"]["mrr25"] - 0.001
        and gen_report["known"]["merged"]["top1"]
        >= gen_report["known"]["baseline"]["top1"] - 0.005
    ):
        rows = generate(
            root,
            split="researchholdout",
            limit=generation_limit,
            samples=generation_samples,
            fragmenter=fragmenter,
        )
        fresh_report = generated_report(root, rows, "researchholdout")
        fresh_report["independent_acceptance"] = False
        fresh_report["note"] = (
            "Bounded pilot subset, not full 4000-molecule performance acceptance."
        )
        write_json(root / "generation/holdout_pilot_report.json", fresh_report)
    reports = {
        "chemical": accept(root, fragmenter),
        "representation": representation_accept(root),
        "generation": gen_report,
        "generation_training": generation["status"],
        "seconds": time.monotonic() - start,
        "kaggle_submitted": False,
        "generation_pilot_only": True,
    }
    write_json(root / "workflow_report.json", reports)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--metfrag-jar", type=Path)
    p.add_argument("--generation-limit", type=int, default=100)
    p.add_argument("--generation-samples", type=int, default=128)
    a = p.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.6)
    print(
        json.dumps(
            run(a.root, a.metfrag_jar, a.generation_limit, a.generation_samples),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
