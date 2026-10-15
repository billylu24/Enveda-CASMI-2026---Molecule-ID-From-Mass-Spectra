def _factorial_entry(scope):
    """Append-only E005–E007 overlay; original v80 functions remain authoritative.

    Embedded function-locally after the complete frozen E004 notebook. This module
    never installs dependencies, changes model scores, or recanonicalizes SMILES.
    """

    import hashlib
    import json
    import math
    from pathlib import Path


    def _finite_coverage(scores, candidates, keys):
        count = 0
        for smi, key in zip(candidates, keys):
            value = scores.get(smi)
            if value is None:
                value = scores.get(key)
            try:
                count += value is not None and math.isfinite(float(value))
            except (TypeError, ValueError):
                pass
        return count


    def rerank_row(scope, mid, original, *, window, ice_fallback, processed):
        """Rebuild final metric fusion from same-run prefusion and engine lists.

        ``original`` is the metric-safe reference BEFORE the frozen final overlay,
        not the E004 final output. The original identity, representative selection,
        same-formula slot rerank and tail fill are reused without modification.
        """
        if window not in (40, 80):
            raise ValueError("Only the prespecified 40/80 candidate windows are supported")
        original = list(original)
        audit = {
            "molecule_id": mid,
            "processed": bool(processed),
            "branch": "placeholder",
            "candidate_count": 0,
            "collapsed_source_votes": 0,
            "ice_finite_count": 0,
            "gl_finite_count": 0,
            "fallback_changed": False,
            "fallback_changed_top1": False,
        }
        if not processed:
            return original, audit
        audit["branch"] = "no_engine"
        engine = scope["ENG"].get(mid)
        if mid not in scope["PREFUSION_LISTS"] or not engine or not engine.get("smiles"):
            return original, audit
        first = [s for s in scope["PREFUSION_LISTS"][mid] if s and s != "CCO"]
        candidates, scores, collapsed = scope["metric_fusion"](
            first,
            engine["smiles"],
            scope["metric_identity"],
            alpha=scope["ALPHA"],
            offset=scope["KRR"],
            limit=window,
        )
        keys = [scope["chem"].score_key(s) or s for s in candidates]
        ice_scores = scope["ICE_SCORES"]
        ice = ice_scores.get(mid, {}) if isinstance(ice_scores, dict) else {}
        gl = scope["gl_of"](mid)
        audit.update(
            branch="unrescored",
            candidate_count=len(candidates),
            collapsed_source_votes=collapsed,
            ice_finite_count=_finite_coverage(ice, candidates, keys),
            gl_finite_count=_finite_coverage(gl, candidates, keys),
        )
        before = list(candidates)
        if ice and gl:
            # Preserve v80's exact GL-present gate and GL-only score arguments.
            order = scope["gl_fuse"].rerank_multi(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                [{}, gl],
                [scope["ICE_LAM"], scope["GL_LAM"]],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "glacier"
        elif (ice_fallback and not gl and scope.get("FRAG_GAP", {}).get(mid)
              and not any(v is not None for v in (ice or {}).values())):
            # v7c: no forward-model evidence at all -> MetFrag-lite gap re-rank (lambda 0.5), once
            order = scope["gl_fuse"].rerank_multi(
                candidates, keys, scores, [scope["_form2"](s) for s in candidates],
                [scope["FRAG_GAP"][mid]], [0.5], top_n=len(candidates))
            candidates = [candidates[i] for i in order]
            audit["branch"] = "frag_gap"
        elif ice_fallback and ice and not gl:
            # Public v1 / script 356785434: ICE is used only where gl_of(mid) is empty.
            # Keep its truthy-dict gate: all-None ICE is an observable harmless no-op.
            order = scope["ice_fuse"].rerank(
                candidates,
                keys,
                scores,
                [scope["_form2"](s) for s in candidates],
                ice,
                lam=scope["ICE_LAM"],
                top_n=len(candidates),
            )
            candidates = [candidates[i] for i in order]
            audit["branch"] = "ice_fallback"
        result = (
            scope["safe_extend"](candidates[:25], candidates[25:] + original)
            if candidates
            else original
        )
        if audit["branch"] == "ice_fallback":
            without_ice = (
                scope["safe_extend"](before[:25], before[25:] + original) if before else original
            )
            audit["fallback_changed"] = result != without_ice
            audit["fallback_changed_top1"] = result[:1] != without_ice[:1]
        return result, audit


    def run_overlay(scope, *, experiment, window, ice_fallback, provenance, workdir="."):
        """Save the E004 bytes, validate same-run control, then emit one treatment."""
        import pandas as pd

        root = Path(workdir)
        path = root / "submission.csv"
        control_bytes = path.read_bytes()
        control_sha = hashlib.sha256(control_bytes).hexdigest()
        base_report = json.loads((root / "e004_verification.json").read_text())
        assert base_report["experiment"] == "E004"
        assert base_report["hashes"]["submission.csv"] == control_sha
        # Copy bytes directly; never reserialize or overwrite original E004 audit files.
        (root / "submission_e004_control.csv").write_bytes(control_bytes)
        control = pd.read_csv(path, dtype=str, keep_default_na=False)
        reference = pd.read_csv(root / "submission_reference.csv", dtype=str, keep_default_na=False)
        assert list(reference.columns) == list(control.columns) == ["molecule_id", "smiles"]
        assert list(reference.molecule_id) == list(control.molecule_id)
        assert control.molecule_id.is_unique
        processed = set(base_report["processed_ids"])
        assert processed == set(scope["te"].molecule_id.astype(str))
        controls = dict(zip(control.molecule_id, control.smiles))
        output, audits = [], []
        for row in reference.itertuples(index=False):
            mid, original = str(row.molecule_id), str(row.smiles).split(";")
            active = mid in processed
            # The exact original pre-overlay tail is essential for paired equivalence.
            rebuilt, _ = rerank_row(
                scope, mid, original, window=40, ice_fallback=False, processed=active
            )
            assert ";".join(rebuilt) == controls[mid], f"E004 control reconstruction mismatch: {mid}"
            result, audit = rerank_row(
                scope, mid, original, window=window, ice_fallback=ice_fallback, processed=active
            )
            assert 1 <= len(result) <= 25
            if active:
                identities = [scope["metric_identity"](s) for s in result]
                assert None not in identities and len(identities) == len(set(identities)), mid
            else:
                assert ";".join(result) == controls[mid], f"Placeholder changed: {mid}"
            old = controls[mid].split(";")
            audit.update(changed=result != old, changed_top1=result[:1] != old[:1])
            audits.append(audit)
            output.append((mid, ";".join(result)))
        frame = pd.DataFrame(output, columns=["molecule_id", "smiles"])
        assert list(frame.molecule_id) == list(control.molecule_id)
        frame.to_csv(path, index=False)
        report = {
            "experiment": experiment,
            "window": window,
            "ice_fallback": bool(ice_fallback),
            "source_experiment": "E004",
            "base_report_file": "e004_verification.json",
            "control_file": "submission_e004_control.csv",
            "control_sha256": control_sha,
            "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "processed_ids": sorted(processed),
            "output_rows": len(frame),
            "changed_rows": sum(a["changed"] for a in audits),
            "changed_top1": sum(a["changed_top1"] for a in audits),
            "fallback_rows": sum(a["branch"] == "ice_fallback" for a in audits),
            "fallback_finite_rows": sum(
                a["branch"] == "ice_fallback" and a["ice_finite_count"] > 0 for a in audits
            ),
            "fallback_changed_rows": sum(a["fallback_changed"] for a in audits),
            "fallback_changed_top1": sum(a["fallback_changed_top1"] for a in audits),
            "same_run_control_reconstructed": True,
            "provenance": provenance,
            "limitations": [
                "Visible smoke is not independent validation.",
                "An inactive fallback in visible smoke says nothing about hidden benefit.",
            ],
        }
        pd.DataFrame(audits).to_csv(root / "experiment_audit_factorial.csv", index=False)
        (root / "experiment_audit_factorial.json").write_text(
            json.dumps(dict(report, rows=audits), indent=2, allow_nan=False) + "\n"
        )
        (root / "experiment_verification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print("FACTORIAL_VERIFICATION", json.dumps(report))
        return report

    return run_overlay(scope, **{'experiment': 'E006', 'window': 40, 'ice_fallback': True, 'provenance': {'base_notebook_sha256': '1f0f9b3026a93ddb5ad6fa6993c8ac0a4bfaddc54594eb1dd6dda7a20d8ef0d6', 'base_source_code_sha256': 'eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2', 'helper_source_sha256': 'a027a69ddda2c2e14639e3ec5196d5f1eb2ab57cba10ef929723148f31fc8f34', 'source_script_version_id': 356740494, 'ice_policy_source_script_version_id': 356785434, 'ice_policy_source': 'samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438/1'}})

FACTORIAL_REPORT = _factorial_entry(globals())
del _factorial_entry
