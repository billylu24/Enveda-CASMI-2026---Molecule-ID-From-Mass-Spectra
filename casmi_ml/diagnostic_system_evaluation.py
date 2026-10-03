"""Local full181 comparison using observable actual deployment logic and fixed baseline."""

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

from casmi_ml.chembl_routed_inference import extend
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.metfrag import digest
from casmi_ml.ranking import ReferenceIndex, metrics
from casmi_ml.research_protocol import ROOT, freeze

BASE = Path("artifacts/research_loop/rounds/0149_chembl_routed_combination")
BEST = Path("artifacts/research_loop/rounds/0181_high_strong_slot_full")


def validate_residual_binding(encoder, original):
    """Critic rebinding is valid only when every original encoder tensor is frozen."""
    base = torch.load(original, map_location="cpu", weights_only=True)
    residual = torch.load(encoder, map_location="cpu", weights_only=True)
    metadata = residual.get("diagnostic_residual", {})
    if (
        metadata.get("base_sha256") != digest(original)
        or metadata.get("dimension") != 24
    ):
        raise ValueError("Diagnostic residual base binding differs")
    original_state, state = base["state_dict"], residual["state_dict"]
    expected = {"base." + k for k in original_state}
    if {k for k in state if k.startswith("base.")} != expected:
        raise ValueError("Diagnostic residual base tensor keys differ")
    for key, value in original_state.items():
        if not torch.equal(value, state["base." + key]):
            raise ValueError("Diagnostic residual changed frozen base tensor: " + key)
    prep = dict(residual["preprocessing"])
    precision = prep.pop("diagnostic_precision", None)
    if prep != base["preprocessing"] or precision is not metadata.get("precise"):
        raise ValueError("Diagnostic residual preprocessing differs")
    for key in ("architecture", "metadata_dim"):
        if residual[key] != base[key]:
            raise ValueError("Diagnostic residual architecture differs")
    return {
        "base_sha256": digest(original),
        "base_tensor_count": len(original_state),
        "all_base_tensors_exact": True,
    }


def run(
    output,
    encoder,
    proposals,
    control=False,
    audit_control=False,
    control_reference=None,
    reuse_unknown=None,
    paired_control=None,
):
    if control and audit_control:
        raise ValueError("Strict control and audit-only modes are mutually exclusive")
    comparison = BEST
    if control_reference is not None:
        if not control:
            raise ValueError("Audited reference is only permitted for strict control")
        comparison = Path(control_reference)
        audit = json.loads((comparison / "report.json").read_text())
        if not audit.get("control_passed"):
            raise ValueError("Independent control attribution did not pass")
    paired = Path(paired_control) if paired_control else None
    if paired:
        paired_protocol = json.loads((paired / "protocol.json").read_text())
        if not paired_protocol.get("control") or not paired_protocol.get(
            "control_reference"
        ):
            raise ValueError(
                "Paired comparison must be independently audited strict control"
            )
        json.loads((paired / "report.json").read_text())
    output, encoder, proposals = Path(output), Path(encoder), Path(proposals)
    output.mkdir(parents=True, exist_ok=True)
    if reuse_unknown and not (control or audit_control):
        raise ValueError("Replay reuse is only permitted for original control")
    reuse = Path(reuse_unknown) if reuse_unknown else None
    config = json.loads((BEST / "external_config.json").read_text())
    base_binding = (
        validate_residual_binding(encoder, config["encoder"]) if paired else None
    )
    original_critic = Path(config["critic"])
    weights = torch.load(original_critic, map_location="cpu", weights_only=True)
    weights["encoder_sha256"] = digest(encoder)
    critic = output / "critic_binding.pt"
    if not critic.exists():
        torch.save(weights, critic)
    config.update(
        encoder=str(encoder),
        encoder_sha256=digest(encoder),
        critic=str(critic),
        critic_sha256=digest(critic),
    )
    freeze(
        output / "protocol.json",
        {
            "critic_frozen_base_verification": base_binding,
            "control": control,
            "audit_control": audit_control,
            "control_reference": str(comparison),
            "control_reference_sha256": digest(comparison / "selected_rankings.json"),
            "paired_control_report_sha256": digest(paired / "report.json")
            if paired
            else None,
            "paired_control_rankings_sha256": digest(paired / "selected_rankings.json")
            if paired
            else None,
            "reused_unknown_outputs": {
                name: digest(reuse / "unknown" / name)
                for name in ("full.json", "selected.csv.external.report.json")
            }
            if reuse
            else None,
            "source_sha256": digest(__file__),
            "inference_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "encoder_sha256": digest(encoder),
            "critic_original_sha256": digest(original_critic),
            "config": config,
            "incumbent_report_sha256": digest(BEST / "report.json"),
            "baseline_full_sha256": digest(BASE / "low_rankings.json"),
            "proposal_sha256": digest(proposals),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "rule": "Freeze0062 full baseline and exact181 observable rule; only residual fingerprint prediction changes candidate preselection. Critic latent, original generation, references and scientific gates frozen.",
            "cohort": "repeated_development",
            "independent_acceptance": False,
            "holdout_used": False,
            "resource_scope": "Cached full local development accuracy evaluation, not cold inference proof",
            "empty_baseline_policy": "Five original empty unknown rankings remain empty; excluded only from CSV validator, restored before full2000 metrics.",
        },
    )
    originals = json.loads((BASE / "low_rankings.json").read_text())["baseline"]
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    generated = {
        r["key"]: {x["key"]: x["smiles"] for x in r["candidates"]}
        for r in json.loads(GENERATED.read_text())
    }
    proposal_data = json.loads(proposals.read_text())
    report, selected, execution = {}, {}, {}
    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    for mode in ["unknown", "known"]:
        original = originals[mode]
        directory = output / mode
        directory.mkdir(exist_ok=True)
        if mode == "known":
            lookup.update(candidate_lookup(ROOT, "researchdev", mode))
            for row in pd.read_parquet(
                "artifacts/research_loop/rounds/0001_mass_v2/reference/known/rows.parquet",
                columns=["inchikey14", "normalized_smiles"],
            ).itertuples():
                lookup.setdefault(row.inchikey14, row.normalized_smiles)
        for row in pd.read_parquet(
            "external/pubchemlite/structures.parquet",
            columns=["inchikey14", "normalized_smiles"],
        ).itertuples():
            lookup.setdefault(row.inchikey14, row.normalized_smiles)
        full = []
        for key, keys in original.items():
            if not keys:
                continue
            smiles = [generated[key].get(k, lookup.get(k)) for k in keys]
            if any(s is None for s in smiles):
                raise ValueError("Original baseline representation missing")
            full.append({"molecule_id": key, "smiles": smiles})
        ids = {r["molecule_id"] for r in full}
        test = (
            frame[frame.inchikey14.isin(ids)]
            .drop(
                columns=[
                    c
                    for c in ["normalized_smiles", "fingerprint", "molecular_formula"]
                    if c in frame
                ]
            )
            .rename(columns={"inchikey14": "molecule_id"})
        )
        test.to_parquet(directory / "test.parquet", index=False)
        write_json(directory / "baseline_full.json", full)
        pd.DataFrame(
            [
                {"molecule_id": r["molecule_id"], "smiles": ";".join(r["smiles"][:25])}
                for r in full
            ]
        ).to_csv(directory / "baseline.csv", index=False)
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        pd.DataFrame(
            [{"molecule_id": k, "confidence": confidence[k]} for k in ids]
        ).to_csv(directory / "routing.csv", index=False)
        started = time.monotonic()
        if reuse and mode == "unknown":
            previous = json.loads((reuse / "protocol.json").read_text())
            for name in (
                "encoder_sha256",
                "critic_original_sha256",
                "inference_source_sha256",
                "baseline_full_sha256",
                "proposal_sha256",
                "development_sha256",
            ):
                if (
                    previous[name]
                    != json.loads((output / "protocol.json").read_text())[name]
                ):
                    raise ValueError(
                        "Reused original control input or implementation differs: "
                        + name
                    )
            for name in (
                "test.parquet",
                "baseline_full.json",
                "baseline.csv",
            ):
                if digest(reuse / mode / name) != digest(directory / name):
                    raise ValueError("Reused control observable input differs: " + name)
            previous_routing = (
                pd.read_csv(reuse / mode / "routing.csv")
                .sort_values("molecule_id")
                .reset_index(drop=True)
            )
            current_routing = (
                pd.read_csv(directory / "routing.csv")
                .sort_values("molecule_id")
                .reset_index(drop=True)
            )
            if not previous_routing.equals(current_routing):
                raise ValueError(
                    "Reused control routing identities or confidence differ"
                )
            for name in (
                "full.json",
                "selected.csv",
                "selected.csv.external.csv",
                "selected.csv.external.report.json",
            ):
                shutil.copy2(reuse / mode / name, directory / name)
        else:
            extend(
                directory / "test.parquet",
                directory / "baseline.csv",
                directory / "baseline_full.json",
                directory / "routing.csv",
                directory / "selected.csv",
                Path("."),
                config,
                deadline=started + 3600,
                fragment_deadline=started + 1200,
                java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
                cache="artifacts/chemistry_20261001/chembl_metfrag_cache",
                output_full_rankings=directory / "full.json",
                reference=ReferenceIndex(
                    "artifacts/research_loop/rounds/0001_mass_v2/reference/" + mode
                ),
            )
        execution[mode] = json.loads(
            (directory / "selected.csv.external.report.json").read_text()
        )
        ranks = {
            r["molecule_id"]: [
                Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in r["smiles"]
            ]
            for r in json.loads((directory / "full.json").read_text())
        }
        ranks.update({k: [] for k, keys in original.items() if not keys})
        if set(ranks) != set(original):
            raise ValueError("Full cohort keys differ")
        pools = {k: list(set(original[k]) | set(proposal_data[k])) for k in ranks}
        selected[mode] = ranks
        report[mode] = {}
        baseline = json.loads((BEST / "selected_rankings.json").read_text())[mode]
        if control or audit_control:
            expected = json.loads((comparison / "selected_rankings.json").read_text())[
                mode
            ]
            mismatches = [k for k in expected if expected[k] != ranks[k]]
            if mismatches:
                write_json(directory / "private_control_mismatches.json", mismatches)
                if control:
                    raise ValueError(
                        "Full original181 observable inference control differs; inspect before evaluating residual models"
                    )
            if audit_control:
                write_json(
                    directory / "private_control_audit.json",
                    {
                        "cohort": len(ranks),
                        "full_ranking_mismatches": mismatches,
                        "top25_mismatches": [
                            k for k in mismatches if baseline[k][:25] != ranks[k][:25]
                        ],
                        "strict_control_passed": not mismatches,
                    },
                )
        for name, values in [("baseline", baseline), ("diagnostic_residual", ranks)]:
            m, per = metrics(values, pools)
            report[mode][name] = m
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
        print(mode, report[mode], flush=True)
    write_json(output / "selected_rankings.json", selected)
    report.update(
        diagnostic_only=control or audit_control,
        independent_acceptance=False,
        local_only=True,
    )
    report["execution_diagnostics"] = {
        mode: {
            k: value[k]
            for k in ("molecules", "seconds", "parent_peak_rss_mib", "status_counts")
        }
        for mode, value in execution.items()
    }
    budget_fallbacks = sum(
        count
        for value in execution.values()
        for status, count in value["status_counts"].items()
        if "budget" in status
    )
    report["development_budget_fallbacks"] = budget_fallbacks
    if paired and budget_fallbacks:
        report["diagnostic_only"] = True
        report["promotion_blocked_reason"] = (
            "Incomplete physical evidence under cached development deadline; repeat bounded run after cache preparation before accuracy selection"
        )
    if paired:
        paired_report = json.loads((paired / "report.json").read_text())
        report["required_comparators"] = {
            "completed_evidence_original_model": {
                "directory": str(paired),
                "variant": "diagnostic_residual",
                "metrics": {
                    mode: paired_report[mode]["diagnostic_residual"]
                    for mode in ("unknown", "known")
                },
            }
        }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--proposals", type=Path, required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--control", action="store_true")
    modes.add_argument("--audit-control", action="store_true")
    p.add_argument("--control-reference", type=Path)
    p.add_argument("--reuse-unknown", type=Path)
    p.add_argument("--paired-control", type=Path)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
