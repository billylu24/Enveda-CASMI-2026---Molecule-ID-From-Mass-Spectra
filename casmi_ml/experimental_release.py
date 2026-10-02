"""Development-gated packaging, explicitly distinct from independent acceptance."""

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from casmi_ml.chemistry_release import prepare
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import release_identity
from casmi_ml.research_release import copy_inference_source


def package(identifier, decision, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh release directory; never overwrite scored assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output.name + ".preparing-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "release"
        _package(identifier, decision, staged)
        staged.rename(output)
    return output


def _package(identifier, decision, output):
    decision, output = Path(decision), Path(output)
    report = json.loads(decision.read_text())
    winner = report["winner"]
    if not winner or not winner["gate"]["eligible"]:
        raise ValueError("Development gate required")
    if report["direction"] not in [
        "mass",
        "generation_slots",
        "coverage",
        "reference_guard",
        "reference_generation",
        "generation_position",
        "generation_model",
        "generated_frequency",
        "protected_generation",
    ]:
        raise ValueError(
            "This packaging path supports tested mass and generation-slot experiments"
        )
    if output.exists():
        raise ValueError("Use a fresh release directory; never overwrite scored assets")
    # Independently accepted chemistry is the base; the added mass rule is experimental.
    prepare(output=output)
    bundle = output / "bundle"
    copy_inference_source(bundle)
    recipe = json.loads((bundle / "deployment_recipe.json").read_text())
    recipe.update(
        mass_hypothesis=winner["variant"],
        status="development_experimental",
        independent_acceptance=False,
        development_decision="development_decision.json",
    )
    if report["direction"] == "generation_slots":
        from casmi_ml.research_protocol import ENCODER, ROOT

        protocol = json.loads(
            (Path(report["round_directory"]) / "protocol.json").read_text()
        )
        if protocol.get("stable_sampling") != "shared_group_forward_v2":
            raise ValueError(
                "Shared deployable sampler development evaluation required"
            )
        base = json.loads(
            (
                Path(
                    protocol.get(
                        "incumbent_directory", report.get("incumbent_directory")
                    )
                )
                / "protocol.json"
            ).read_text()
        )
        if (
            digest(
                Path(
                    protocol.get(
                        "incumbent_directory", report.get("incumbent_directory")
                    )
                )
                / "report.json"
            )
            != protocol["incumbent_report_sha256"]
        ):
            raise ValueError("Generation incumbent report changed")
        replay = json.loads(
            (Path(report["round_directory"]) / "replay/verification.json").read_text()
        )
        if not replay["valid"] or replay["molecules"] < 25:
            raise ValueError("Real low-confidence generation inference replay required")
        if base.get("metfrag_weight") != 0.5 or base.get("neural_weight") != 0.75:
            raise ValueError("Unexpected generation incumbent")
        _, prefix, slots = winner["variant"].split("_")
        if replay.get("prefix") != int(prefix) or replay.get("slots") != int(slots):
            raise ValueError(
                "Replay configuration differs from selected generator slots"
            )
        recipe["mass_hypothesis"] = protocol["incumbent_variant"]
        checkpoint = Path(
            protocol.get("generator_checkpoint", ROOT / "generation/smiles_42/model.pt")
        )
        if protocol.get("generator_sha256", digest(checkpoint)) != digest(checkpoint):
            raise ValueError("Selected generator checksum changed")
        import torch

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"]["encoder_sha256"] != digest(ENCODER):
            raise ValueError("Generation conditioning encoder changed")
        shutil.copy2(checkpoint, bundle / "generation.pt")
        recipe["generation"] = {
            "checkpoint": "generation.pt",
            "sha256": digest(checkpoint),
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "prefix": int(prefix),
            "slots": int(slots),
            "samples": 128,
            "total_seconds": 1800,
        }
    if report["direction"] == "coverage":
        protocol = json.loads(
            (Path(report["round_directory"]) / "protocol.json").read_text()
        )
        from casmi_ml.coverage_experiment import EXTERNAL

        if digest(EXTERNAL) != protocol["external"]["derived_sha256"]:
            raise ValueError("Selected expansion catalog changed")
        recipe["mass_hypothesis"] = "charge_aware_union"
        shutil.copy2(EXTERNAL, bundle / "external_catalog.parquet")
        attribution = bundle / "catalog_attribution"
        attribution.mkdir(exist_ok=True)
        for name in ["ATTRIBUTION.md", "manifest.json", "zenodo_record.json"]:
            shutil.copy2(EXTERNAL.parent / name, attribution / name)
        recipe["chemistry_expansion"] = {
            "path": "external_catalog.parquet",
            "sha256": digest(EXTERNAL),
            "weight": float(winner["variant"].split("_")[1]),
        }
        replay = json.loads(
            (Path(report["round_directory"]) / "replay.json").read_text()
        )
        if not replay["valid"] or replay["molecules"] < 25:
            raise ValueError("Real expanded chemical ranking replay required")
    if report["direction"] in [
        "reference_guard",
        "reference_generation",
        "generation_position",
        "generation_model",
        "generated_frequency",
    ]:
        from casmi_ml.coverage_experiment import EXTERNAL
        from casmi_ml.research_protocol import ROOT

        protocol = json.loads(
            (Path(report["round_directory"]) / "protocol.json").read_text()
        )
        open_protected = report["direction"] in [
            "reference_generation",
            "generation_position",
            "generation_model",
            "generated_frequency",
        ]
        if protocol.get("open_protected", False) != open_protected:
            raise ValueError("Reference generation protocol mismatch")
        source = Path(protocol["source_directory"])
        if digest(source / "report.json") != protocol["source_report_sha256"]:
            raise ValueError("Reference guard source ranks changed")
        prefix, slots = 5, 5
        checkpoint = ROOT / "generation/smiles_42/model.pt"
        if report["direction"] in ["generation_model", "generated_frequency"]:
            if report["direction"] == "generated_frequency" and (
                winner["variant"] not in protocol["variants"]
                or not 0 < protocol["variants"][winner["variant"]] <= 1
            ):
                raise ValueError("Selected frequency weight required")
            if (
                report["direction"] == "generation_model"
                and winner["variant"] != "model"
            ) or protocol.get("limit") is not None:
                raise ValueError("Full generator checkpoint ranking required")
            prefix, slots = protocol["prefix"], protocol["slots"]
            checkpoint = Path(protocol["generator_checkpoint"])
            if digest(checkpoint) != protocol["generator_sha256"]:
                raise ValueError("Selected decoder checkpoint changed")
        elif report["direction"] == "generation_position":
            if not protocol.get("slot_ablation"):
                raise ValueError("Generation position ablation protocol required")
            _, prefix, slots = winner["variant"].split("_")
            prefix, slots = int(prefix), int(slots)
        elif winner["variant"] != "reference_1_0":
            raise ValueError(
                "Reference guard packaging supports the selected top1 rule"
            )
        replay = json.loads(
            (Path(report["round_directory"]) / "replay.json").read_text()
        )
        if not replay["valid"] or replay["molecules"] < 25:
            raise ValueError("Reference guard branch replay required")
        if open_protected and (
            not replay.get("open_protected")
            or replay.get("high_confidence_branch", 0) < 25
        ):
            raise ValueError("High-confidence reference generation replay required")
        if report["direction"] in [
            "generation_position",
            "generation_model",
            "generated_frequency",
        ] and (replay.get("prefix") != prefix or replay.get("slots") != slots):
            raise ValueError("Generation position replay configuration mismatch")
        source_protocol = json.loads((source / "protocol.json").read_text())
        if digest(EXTERNAL) != source_protocol["external"]["derived_sha256"]:
            raise ValueError("Reference guard external catalog changed")
        suffix = (
            ""
            if checkpoint.resolve()
            == (ROOT / "generation/smiles_42/model.pt").resolve()
            else "_" + digest(checkpoint)[:12]
        )
        generated = (
            ROOT
            / "generation"
            / f"researchdev_samples128_limitall_stable_v2{suffix}.json"
        )
        if report["direction"] == "generated_frequency":
            generated = Path(protocol["generated_path"])
            if (
                replay.get("frequency_weight")
                != protocol["variants"][winner["variant"]]
            ):
                raise ValueError("Frequency replay weight differs from selected rule")
        if report["direction"] in ["generation_model", "generated_frequency"]:
            if replay.get("generator_sha256") != digest(checkpoint) or replay.get(
                "samples_sha256"
            ) != digest(generated):
                raise ValueError("Decoder checkpoint replay evidence changed")
            if len(json.loads(generated.read_text())) != 2000:
                raise ValueError("Full decoder development samples required")
            import torch

            from casmi_ml.research_protocol import ENCODER

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["config"]["encoder_sha256"] != digest(ENCODER):
                raise ValueError("Decoder conditioning encoder changed")
        elif digest(generated) != protocol["generated_sha256"]:
            raise ValueError("Reference guard generator candidates changed")
        generated_config = json.loads(generated.with_suffix(".config.json").read_text())
        if digest(checkpoint) != generated_config["checkpoint_sha256"]:
            raise ValueError("Reference guard generation model changed")
        recipe["mass_hypothesis"] = "charge_aware_union"
        recipe["reference_guard"] = {"topn": 1, "threshold": 0.0}
        shutil.copy2(EXTERNAL, bundle / "external_catalog.parquet")
        recipe["chemistry_expansion"] = {
            "path": "external_catalog.parquet",
            "sha256": digest(EXTERNAL),
            "weight": 1.0,
        }
        attribution = bundle / "catalog_attribution"
        attribution.mkdir(exist_ok=True)
        for name in ["ATTRIBUTION.md", "manifest.json", "zenodo_record.json"]:
            shutil.copy2(EXTERNAL.parent / name, attribution / name)
        shutil.copy2(checkpoint, bundle / "generation.pt")
        recipe["generation"] = {
            "checkpoint": "generation.pt",
            "sha256": digest(checkpoint),
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "prefix": prefix,
            "slots": slots,
            "samples": 128,
            "total_seconds": 1800,
            "open_protected": open_protected,
        }
        if report["direction"] == "generated_frequency":
            if (
                generated_config.get("candidate_statistics")
                != "sample_frequency_and_best_sequence_token_count_v1"
            ):
                raise ValueError("Measured candidate frequency cache required")
            recipe["generation"]["frequency_weight"] = protocol["variants"][
                winner["variant"]
            ]
    if report["direction"] == "protected_generation":
        from casmi_ml.research_protocol import ROOT

        protocol = json.loads(
            (Path(report["round_directory"]) / "protocol.json").read_text()
        )
        generated = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
        mass = Path(
            protocol.get(
                "mass_directory", "artifacts/research_loop/rounds/0001_mass_v2"
            )
        )
        if digest(mass / "report.json") != protocol["mass_report_sha256"]:
            raise ValueError("Protected generation retrieval changed")
        if digest(generated) != protocol["generation_sha256"]:
            raise ValueError("Protected generation samples changed")
        generated_config = json.loads(generated.with_suffix(".config.json").read_text())
        if (
            digest(ROOT / "generation/smiles_42/model.pt")
            != generated_config["checkpoint_sha256"]
        ):
            raise ValueError("Protected generation checkpoint changed")
        _, prefix, slots = winner["variant"].split("_")
        replay = json.loads(
            (Path(report["round_directory"]) / "replay/verification.json").read_text()
        )
        if (
            not replay["valid"]
            or not replay.get("open_protected")
            or replay["protected_queries"] < 25
        ):
            raise ValueError("Actual high-confidence generation replay required")
        if replay["prefix"] != int(prefix) or replay["slots"] != int(slots):
            raise ValueError("Protected generation replay configuration mismatch")
        checkpoint = ROOT / "generation/smiles_42/model.pt"
        recipe["mass_hypothesis"] = "charge_aware_union"
        shutil.copy2(checkpoint, bundle / "generation.pt")
        recipe["generation"] = {
            "checkpoint": "generation.pt",
            "sha256": digest(checkpoint),
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "prefix": int(prefix),
            "slots": int(slots),
            "samples": 128,
            "total_seconds": 1800,
            "open_protected": True,
        }
    write_json(bundle / "deployment_recipe.json", recipe)
    shutil.copy2(decision, bundle / "development_decision.json")
    sums = {
        str(p.relative_to(bundle)): digest(p)
        for p in bundle.rglob("*")
        if p.is_file()
        and p.name not in ["manifest.json", "SHA256SUMS.json"]
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    write_json(bundle / "SHA256SUMS.json", sums)
    with zipfile.ZipFile(
        output / "dataset/casmi_chemistry_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for name in [*sums, "SHA256SUMS.json"]:
            z.write(bundle / name, name)
    slug = identifier.lower().replace("_", "-")
    dataset_id = f"giaok246/casmi26-research-{slug}-bundle"
    kernel_id = f"giaok246/casmi26-research-{slug}"
    dm = json.loads((output / "dataset/dataset-metadata.json").read_text())
    dm.update(
        id=dataset_id,
        title=f"CASMI Research {identifier} Assets",
        description="Private development-gated experimental deployment. Added mass hypotheses have no independent acceptance.",
    )
    write_json(output / "dataset/dataset-metadata.json", dm)
    km = json.loads((output / "notebook/kernel-metadata.json").read_text())
    km.update(
        enable_gpu=report["direction"]
        in [
            "generation_slots",
            "reference_guard",
            "reference_generation",
            "generation_position",
            "generation_model",
            "generated_frequency",
            "protected_generation",
        ],
        id=kernel_id,
        title=f"CASMI26 Research {slug.replace(chr(45), chr(32))}",
        dataset_sources=[dataset_id, "aidensong123/casmi26-coconut-202609"],
    )
    write_json(output / "notebook/kernel-metadata.json", km)
    nb = json.loads((output / "notebook/casmi_chemistry.ipynb").read_text())
    if report["direction"] in [
        "generation_slots",
        "reference_guard",
        "reference_generation",
        "generation_position",
        "generation_model",
        "generated_frequency",
        "protected_generation",
    ]:
        nb["cells"][1]["source"] = nb["cells"][1]["source"].replace(
            "casmi_ml.secondary_inference", "casmi_ml.research_pipeline"
        )
    nb["cells"][0]["source"] = (
        f"# CASMI Research {identifier}\nDevelopment-gated experimental mass hypotheses; repeated development validation, no independent acceptance for this extension.\n"
    )
    write_json(output / "notebook/casmi_chemistry.ipynb", nb)
    identity = release_identity(output, sums, winner["variant"])
    write_json(
        output / "status.json",
        {
            "status": "prepared",
            "identity": identity,
            "development_eligible": True,
            "independent_acceptance": False,
            "uploaded": False,
            "submitted": False,
        },
    )
    return output


def verify(release, report, valid):
    release, report = Path(release), Path(report)
    data = json.loads(report.read_text())
    status = json.loads((release / "status.json").read_text())
    result = {
        "valid": valid,
        "identity": status["identity"],
        "seconds": data["seconds"],
        "peak_rss_mib": data["peak_rss_mib"],
        "molecules": data["molecules"],
        "independent_acceptance": False,
    }
    write_json(release / "verification.json", result)
    return result


def verify_local(release):
    import os
    import subprocess
    import sys

    import pandas as pd

    from casmi_ml.inference import validate_submission

    release = Path(release).resolve()
    bundle = release / "bundle"
    repo = Path.cwd()
    java = list((repo / "external/metfrag/java21").glob("*/bin/java"))
    if len(java) != 1:
        raise ValueError("Pinned Java 21 required")
    env = dict(
        os.environ,
        PYTHONPATH=str(bundle),
        PATH=str(java[0].parent) + os.pathsep + os.environ["PATH"],
    )
    output = release / "local_output"
    output.mkdir(exist_ok=True)
    submission = output / "submission.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "casmi_ml.research_pipeline",
            "--recipe",
            str(bundle / "deployment_recipe.json"),
            "--data-dir",
            str(repo / "data"),
            "--coconut",
            str(repo / "external/coconut_structures.parquet"),
            "--output",
            str(submission),
        ],
        cwd=bundle,
        env=env,
        check=True,
    )
    validate_submission(
        pd.read_parquet(repo / "data/test.parquet"), pd.read_csv(submission)
    )
    return verify(release, str(submission) + ".report.json", True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round")
    p.add_argument("--decision", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    if a.verify:
        print(verify_local(a.output))
    else:
        if not a.round or not a.decision:
            p.error("--round and --decision required to package")
        print(package(a.round, a.decision, a.output))


if __name__ == "__main__":
    main()
