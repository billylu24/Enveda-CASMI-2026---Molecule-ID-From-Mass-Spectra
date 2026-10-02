"""Unlabeled spectrum-conditioned generation merged into a frozen retrieval CSV."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from casmi_ml.chemistry import extract_evidence, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.generation_experiment import load_model, validate_generated
from casmi_ml.inference import validate_submission
from casmi_ml.metfrag import digest
from casmi_ml.ranking import rrf
from casmi_ml.research_protocol import ENCODER
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def validated_confidence(values):
    # Cosine scores can overshoot unit bounds by float32 rounding error.
    tolerance = 1e-6
    if any(
        not np.isfinite(v) or not -tolerance <= v <= 1 + tolerance
        for v in values.values()
    ):
        raise ValueError("Finite retrieval confidence between 0 and 1 required")
    return {key: float(np.clip(value, 0, 1)) for key, value in values.items()}


@torch.inference_mode()
def predict(
    checkpoint,
    test_path,
    baseline_csv,
    output,
    routing_csv=None,
    samples=128,
    seconds=1800,
    encoder_path=None,
    prefix=None,
    slots=None,
    full_rankings=None,
    open_protected=False,
    frequency_weight=0.0,
    token_length_exponent=0.0,
    critic_checkpoint=None,
    critic_weight=0.0,
    adaptive_prefix=None,
    expanded_prefix=None,
    first_gate=None,
):
    if not 0 <= frequency_weight <= 1:
        raise ValueError("Frequency fusion weight must be between zero and one")
    if not 0 <= token_length_exponent <= 1 or not 0 <= critic_weight <= 1:
        raise ValueError("Invalid generated calibration weights")
    if critic_weight and critic_checkpoint is None:
        raise ValueError("Critic weights require a frozen critic checkpoint")
    if expanded_prefix is not None and expanded_prefix not in (2, 3, 5, 10):
        raise ValueError("Unsupported expanded generation prefix")
    if first_gate is not None and (
        set(first_gate) != {"confidence", "margin"}
        or not np.isfinite(list(first_gate.values())).all()
        or not 0 < first_gate["confidence"] <= 0.5
        or not 0 <= first_gate["margin"] <= 1
        or not critic_weight
    ):
        raise ValueError(
            "First-generation gate requires finite thresholds and frozen critic"
        )
    encoder_path = Path(encoder_path or ENCODER)
    if not 1 <= samples <= 128 or seconds <= 0:
        raise ValueError("Invalid generation resource limits")
    if prefix is not None and (
        slots is None or prefix < 1 or slots < 1 or full_rankings is None
    ):
        raise ValueError(
            "Explicit slots require positive values and full retrieval rankings"
        )
    configure(threads=4)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"]["encoder_sha256"] != digest(encoder_path):
        raise ValueError("Generation conditioner checksum mismatch")
    decoder, formula, vocabulary = load_model(checkpoint, device)
    encoder, encoder_checkpoint = load_deployment_checkpoint(encoder_path, "scale")
    encoder.to(device)
    critic, critic_encoder = None, None
    if critic_weight:
        from casmi_ml.direct_models import DirectRanker

        critic_saved = torch.load(
            critic_checkpoint, map_location="cpu", weights_only=True
        )
        if (
            critic_saved["encoder_sha256"] != digest(encoder_path)
            or critic_saved["architecture"] != "fingerprint"
        ):
            raise ValueError("Generated critic encoder or architecture mismatch")
        critic = DirectRanker("fingerprint").eval()
        critic.load_state_dict(critic_saved["state_dict"])
        critic_encoder, _ = load_deployment_checkpoint(encoder_path, "scale")
        critic_encoder.eval()
    test = pd.read_parquet(test_path)
    base = pd.read_csv(baseline_csv)
    validate_submission(test, base)
    base = base.set_index("molecule_id")
    full = None
    if full_rankings is not None:
        raw = json.loads(Path(full_rankings).read_text())
        full = {r["molecule_id"]: r["smiles"] for r in raw}
        if len(full) != len(raw) or set(full) != set(test.molecule_id):
            raise ValueError("Full retrieval ranking IDs differ")
        for molecule_id, values in full.items():
            if values[:25] != base.loc[molecule_id, "smiles"].split(";"):
                raise ValueError("Full rankings do not match retrieval submission")
    routing = pd.read_csv(routing_csv or str(baseline_csv) + ".routing.csv")
    if routing.molecule_id.duplicated().any() or set(routing.molecule_id) != set(
        test.molecule_id
    ):
        raise ValueError("Routing IDs do not match test molecules")
    protected = routing.set_index("molecule_id").protected.to_dict()
    if any(not isinstance(v, (bool, np.bool_)) for v in protected.values()):
        raise ValueError("Routing protection must contain booleans")
    if open_protected:
        protected = {key: False for key in protected}
    allowed = {key: True for key in protected}
    if expanded_prefix is not None and "generation_allowed" not in routing:
        raise ValueError("Actual retrieval branch flags required")
    if "generation_allowed" in routing:
        allowed = routing.set_index("molecule_id").generation_allowed.to_dict()
        if any(not isinstance(v, (bool, np.bool_)) for v in allowed.values()):
            raise ValueError("Generation gates must be booleans")
        if expanded_prefix is None:
            protected = {key: protected[key] or not allowed[key] for key in protected}
    confidence = None
    if first_gate is not None:
        if "confidence" not in routing:
            raise ValueError("Actual retrieval confidence required for promotion gate")
        confidence = validated_confidence(
            routing.set_index("molecule_id").confidence.to_dict()
        )
    second_reference = None
    if adaptive_prefix is not None:
        if adaptive_prefix != "second_unreferenced" or prefix != 2:
            raise ValueError("Unsupported adaptive generated prefix")
        if "second_candidate_has_reference" not in routing:
            raise ValueError("Actual second candidate reference membership required")
        second_reference = routing.set_index(
            "molecule_id"
        ).second_candidate_has_reference.to_dict()
        if any(not isinstance(v, (bool, np.bool_)) for v in second_reference.values()):
            raise ValueError("Second candidate reference flags must be boolean")
    rows, audit = [], []
    for molecule_id, group in test.groupby("molecule_id", sort=False):
        smiles = (
            full[molecule_id]
            if full is not None
            else base.loc[molecule_id, "smiles"].split(";")
        )[:]
        original_top25 = smiles[:25]
        status = "protected"
        stats = {}
        if not protected[molecule_id] and time.monotonic() - started < seconds:
            from casmi_ml.generation_sampling import condition_for_group

            condition = condition_for_group(
                encoder,
                encoder_checkpoint["preprocessing"],
                group,
                device,
                saved["config"].get("neutral_mass_condition", False),
            )
            hypotheses = formula.hypotheses(condition)
            # Seed each molecule independently of group processing/fallback order.
            from casmi_ml.generation_sampling import sampling_seed

            seed = sampling_seed(group)
            generator = torch.Generator(device=device).manual_seed(seed)
            try:
                sequence, logp, finished = decoder.generate(
                    torch.cat([condition, formula.soft(condition)], 1),
                    samples,
                    generator=generator,
                    deadline=started + seconds,
                )
            except TimeoutError:
                rows.append(
                    {"molecule_id": molecule_id, "smiles": ";".join(original_top25)}
                )
                audit.append(
                    {"molecule_id": molecule_id, "status": "budget_retrieval_fallback"}
                )
                continue
            masses = [
                m
                for r in group.to_dict("records")
                if (m := neutral_mass(r)) is not None
            ]
            mass = float(np.median(masses)) if masses else None
            evidence = [extract_evidence(r) for r in group.to_dict("records")]
            candidates, stats = validate_generated(
                sequence.cpu().tolist(),
                logp.cpu().tolist(),
                finished.cpu().tolist(),
                vocabulary,
                mass,
                hypotheses,
                evidence,
                track_frequency=frequency_weight > 0 or token_length_exponent > 0,
            )
            if candidates:
                if token_length_exponent or critic_weight:
                    from casmi_ml.generated_score_combination import combined_order

                    scores = {}
                    if critic_weight and len(candidates) > 1:
                        from casmi_ml.data import fingerprint
                        from casmi_ml.direct_models import score_group

                        pool = pd.DataFrame(
                            {"normalized_smiles": [c["smiles"] for c in candidates]}
                        )
                        fps = np.stack(
                            [fingerprint(c["smiles"]) for c in candidates]
                        ).astype(np.float32)
                        values = score_group(
                            critic,
                            critic_encoder,
                            group,
                            encoder_checkpoint["preprocessing"],
                            pool,
                            fps,
                        )
                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite generated critic scores")
                        scores = {
                            c["key"]: float(value)
                            for c, value in zip(candidates, values)
                        }
                    by_key = {c["key"]: c for c in candidates}
                    candidates = [
                        by_key[k]
                        for k in combined_order(
                            candidates,
                            scores,
                            (token_length_exponent, frequency_weight, critic_weight),
                        )
                    ]
                elif frequency_weight:
                    from casmi_ml.generation_frequency_ranking import ranked_candidates

                    by_key = {c["key"]: c for c in candidates}
                    candidates = [
                        by_key[k]
                        for k in ranked_candidates(candidates, frequency_weight)
                    ]
                lookup = {
                    Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]: s for s in smiles
                }
                original = list(lookup)
                generated = [c["key"] for c in candidates]
                lookup.update(
                    {
                        c["key"]: c["smiles"]
                        for c in candidates
                        if c["key"] not in lookup
                    }
                )
                if prefix is None:
                    rank = rrf([original, generated], [0.75, 0.25])
                else:
                    from casmi_ml.generation_slots import insert_generated

                    effective_prefix = (
                        1
                        if second_reference is not None
                        and not second_reference[molecule_id]
                        and len(original) >= 2
                        else prefix
                    )
                    if expanded_prefix is not None and not allowed[molecule_id]:
                        effective_prefix = expanded_prefix
                    if (
                        first_gate is not None
                        and confidence[molecule_id] < first_gate["confidence"]
                    ):
                        from casmi_ml.generated_first_critic import promotion_prefix

                        novel = [c for c in candidates if c["key"] not in set(original)]
                        if original and novel:
                            from casmi_ml.data import fingerprint
                            from casmi_ml.direct_models import score_group

                            pair = [lookup[original[0]], novel[0]["smiles"]]
                            values = score_group(
                                critic,
                                critic_encoder,
                                group,
                                encoder_checkpoint["preprocessing"],
                                pd.DataFrame({"normalized_smiles": pair}),
                                np.stack([fingerprint(s) for s in pair]).astype(
                                    np.float32
                                ),
                            )
                            if not np.isfinite(values).all():
                                raise ValueError("Nonfinite promotion critic scores")
                            effective_prefix = promotion_prefix(
                                original,
                                generated,
                                {
                                    original[0]: float(values[0]),
                                    novel[0]["key"]: float(values[1]),
                                },
                                effective_prefix,
                                first_gate["margin"],
                            )
                        stats["generated_first_promotion"] = effective_prefix == 0
                    rank = insert_generated(
                        original, generated, effective_prefix, slots
                    )
                smiles = [lookup[k] for k in rank[:25]]
                status = "generated_and_merged"
            else:
                status = "no_valid_mass_matching_generation"
        elif not protected[molecule_id]:
            status = "budget_retrieval_fallback"
        smiles = smiles[:25] if status == "generated_and_merged" else original_top25
        rows.append({"molecule_id": molecule_id, "smiles": ";".join(smiles)})
        audit.append({"molecule_id": molecule_id, "status": status, **stats})
    submission = pd.DataFrame(rows)
    validate_submission(test, submission)
    submission.to_csv(output, index=False)
    pd.DataFrame(audit).to_csv(str(output) + ".generation.csv", index=False)
    write_json(
        str(output) + ".report.json",
        {
            "molecules": len(rows),
            "seconds": time.monotonic() - started,
            "checkpoint_sha256": digest(checkpoint),
            "baseline_sha256": digest(baseline_csv),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "formula_oracle_used": False,
            "kaggle_submitted": False,
            "sampling": "label_independent_spectrum_hash_v1",
            "open_protected": open_protected,
            "prefix": prefix,
            "slots": slots,
            "frequency_weight": frequency_weight,
            "token_length_exponent": token_length_exponent,
            "critic_weight": critic_weight,
            "adaptive_prefix": adaptive_prefix,
            "expanded_prefix": expanded_prefix,
            "first_gate": first_gate,
            "critic_checkpoint_sha256": digest(critic_checkpoint)
            if critic_weight
            else None,
        },
    )
    return submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test", type=Path, default=Path("data/test.parquet"))
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--routing", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seconds", type=float, default=1800)
    p.add_argument("--encoder", type=Path, default=ENCODER)
    p.add_argument("--prefix", type=int)
    p.add_argument("--slots", type=int)
    p.add_argument("--full-rankings", type=Path)
    p.add_argument("--open-protected", action="store_true")
    p.add_argument("--frequency-weight", type=float, default=0.0)
    p.add_argument("--token-length-exponent", type=float, default=0.0)
    p.add_argument("--critic-checkpoint", type=Path)
    p.add_argument("--critic-weight", type=float, default=0.0)
    p.add_argument("--adaptive-prefix", choices=["second_unreferenced"])
    p.add_argument("--expanded-prefix", type=int, choices=[2, 3, 5, 10])
    a = p.parse_args()
    if (a.prefix is None) != (a.slots is None):
        p.error("--prefix and --slots must be provided together")
    predict(
        a.checkpoint,
        a.test,
        a.baseline,
        a.output,
        a.routing,
        a.samples,
        a.seconds,
        a.encoder,
        a.prefix,
        a.slots,
        a.full_rankings,
        a.open_protected,
        a.frequency_weight,
        a.token_length_exponent,
        a.critic_checkpoint,
        a.critic_weight,
        a.adaptive_prefix,
        a.expanded_prefix,
    )


if __name__ == "__main__":
    main()
