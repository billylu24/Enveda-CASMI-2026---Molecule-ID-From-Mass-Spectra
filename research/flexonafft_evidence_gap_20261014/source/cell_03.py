'''2. Configuration, credited adapters and routing objects
The adapters execute preserved source cells in one explicit per-experiment namespace to keep upstream imports and subprocess contracts intact. Kernels are not hidden claims of original authorship. Classes own the new routing and audit boundaries.'''

from dataclasses import dataclass
from pathlib import Path
import hashlib, json, math, time
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ExperimentConfig:
    group_fragment_fallback: bool = False
    window: int = 40
    fragment_weight: float = 0.5

class SourceStage:
    def __init__(self, label, source_cells):
        self.label, self.source_cells = label, source_cells

    def run(self, scope):
        started = time.time()
        print('STAGE:', self.label, flush=True)
        for index, source in self.source_cells:
            exec(compile(source, f'<credited-v7c-cell-{index}>', 'exec'), scope)
        return dict(stage=self.label, seconds=round(time.time() - started, 2))

class InputRegistry(SourceStage):
    def run(self, scope):
        import sys
        if sys.version_info[:2] not in ((3, 12), (3, 13)):
            raise RuntimeError(
                f'Supported runtimes: Python 3.12/3.13; got {sys.version_info.major}.{sys.version_info.minor}. '
                'Use a compatible runtime and RDKit 2025.3.6 simulator wheel; never substitute RDKit 2026.')
        return super().run(scope)

class CandidateRetriever(SourceStage):
    pass

class ForwardEvidence(SourceStage):
    pass

class MetricAudit(SourceStage):
    pass

class FragmentEvidenceScorer(SourceStage):
    def __init__(self, label, source_cells):
        super().__init__(label, source_cells)
        self.cache, self.errors = {}, []

    def score(self, scope, mid, candidates):
        module = scope.get('frag_rescore')
        if module is None or '_fcache' not in scope:
            return {}
        frame = scope['te'][scope['te'].molecule_id.astype(str) == str(mid)]
        spectra = [dict(mz=np.asarray(r.ms2_mzs, np.float64),
            it=np.asarray(r.ms2_normalized_intensities, np.float64), adduct=r.adduct,
            prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
            for r in frame.itertuples(index=False)][:16]
        if not spectra:
            return {}
        try:
            values = module.score_molecule(spectra, candidates, cache=self.cache)
            return {s: float(v) for s, v in zip(candidates, values) if v is not None and math.isfinite(float(v))}
        except Exception as error:
            self.errors.append(dict(molecule_id=str(mid), error=repr(error)))
            print('GROUP FRAGMENT FAILED: preserve v7c order', mid, repr(error))
            return {}

class MetricIdentityFusion:
    def __init__(self, scope):
        self.scope = scope

    def rank(self, first, second, window):
        return self.scope['metric_fusion'](first, second, self.scope['metric_identity'],
            alpha=self.scope['ALPHA'], offset=self.scope['KRR'], limit=window)

class EvidenceRouter:
    def __init__(self, scope, config, fragments):
        self.scope, self.config, self.fragments = scope, config, fragments
        self.fusion = MetricIdentityFusion(scope)

    @staticmethod
    def coverage(scores, candidates, keys):
        count = 0
        for smile, key in zip(candidates, keys):
            value = scores.get(smile)
            if value is None:
                value = scores.get(key)
            try:
                count += value is not None and math.isfinite(float(value))
            except (TypeError, ValueError):
                pass
        return count

    def group_rescore(self, mid, candidates, original_candidates, scores, ice, gl):
        scope = self.scope
        keys = [scope['chem'].score_key(s) or s for s in candidates]
        formulas = [scope['_form2'](s) for s in candidates]
        groups = {}
        for index, formula in enumerate(formulas):
            groups.setdefault(formula, []).append(index)
        missing = []
        for indices in groups.values():
            smiles = [candidates[i] for i in indices]
            group_keys = [keys[i] for i in indices]
            if len(indices) >= 2 and not self.coverage(ice, smiles, group_keys) and not self.coverage(gl, smiles, group_keys):
                missing.append(indices)
        if not missing:
            return list(candidates), 0
        needed = [candidates[i] for indices in missing for i in indices]
        evidence = self.fragments.score(scope, mid, needed)
        accepted = {}
        for indices in missing:
            if all(candidates[i] in evidence for i in indices):
                accepted.update((candidates[i], evidence[candidates[i]]) for i in indices)
        if not accepted:
            return list(candidates), 0
        score_by_smile = dict(zip(original_candidates, scores))
        aligned_scores = [score_by_smile[s] for s in candidates]
        order = scope['gl_fuse'].rerank_multi(candidates, keys, aligned_scores, formulas,
            [accepted], [self.config.fragment_weight], top_n=len(candidates))
        # Group-slot preservation is a trust boundary, not just a plotting assumption.
        eligible_positions = {i for indices in missing if all(candidates[j] in accepted for j in indices) for i in indices}
        assert all(i == j for i, j in enumerate(order) if i not in eligible_positions)
        return [candidates[i] for i in order], sum(all(candidates[i] in accepted for i in indices) for indices in missing)
    def rerank_row(self, mid, original, *, window, ice_fallback, processed, group=False):
        scope = self.scope
        'Rebuild final metric fusion from same-run prefusion and engine lists.\n\n        ``original`` is the metric-safe reference BEFORE the frozen final overlay,\n        not the E004 final output. The original identity, representative selection,\n        same-formula slot rerank and tail fill are reused without modification.\n        '
        if window not in (40, 80):
            raise ValueError('Only the prespecified 40/80 candidate windows are supported')
        original = list(original)
        audit = {'molecule_id': mid, 'processed': bool(processed), 'branch': 'placeholder', 'candidate_count': 0, 'collapsed_source_votes': 0, 'ice_finite_count': 0, 'gl_finite_count': 0, 'fallback_changed': False, 'fallback_changed_top1': False, 'group_fragment_groups': 0}
        if not processed:
            return (original, audit)
        audit['branch'] = 'no_engine'
        engine = scope['ENG'].get(mid)
        if mid not in scope['PREFUSION_LISTS'] or not engine or (not engine.get('smiles')):
            return (original, audit)
        first = [s for s in scope['PREFUSION_LISTS'][mid] if s and s != 'CCO']
        candidates, scores, collapsed = self.fusion.rank(first, engine['smiles'], window)
        keys = [scope['chem'].score_key(s) or s for s in candidates]
        ice_scores = scope['ICE_SCORES']
        ice = ice_scores.get(mid, {}) if isinstance(ice_scores, dict) else {}
        gl = scope['gl_of'](mid)
        audit.update(branch='unrescored', candidate_count=len(candidates), collapsed_source_votes=collapsed, ice_finite_count=self.coverage(ice, candidates, keys), gl_finite_count=self.coverage(gl, candidates, keys))
        before = list(candidates)
        if ice and gl:
            order = scope['gl_fuse'].rerank_multi(candidates, keys, scores, [scope['_form2'](s) for s in candidates], [{}, gl], [scope['ICE_LAM'], scope['GL_LAM']], top_n=len(candidates))
            candidates = [candidates[i] for i in order]
            audit['branch'] = 'glacier'
        elif ice_fallback and (not gl) and scope.get('FRAG_GAP', {}).get(mid) and (not any((v is not None for v in (ice or {}).values()))):
            order = scope['gl_fuse'].rerank_multi(candidates, keys, scores, [scope['_form2'](s) for s in candidates], [scope['FRAG_GAP'][mid]], [0.5], top_n=len(candidates))
            candidates = [candidates[i] for i in order]
            audit['branch'] = 'frag_gap'
        elif ice_fallback and ice and (not gl):
            order = scope['ice_fuse'].rerank(candidates, keys, scores, [scope['_form2'](s) for s in candidates], ice, lam=scope['ICE_LAM'], top_n=len(candidates))
            candidates = [candidates[i] for i in order]
            audit['branch'] = 'ice_fallback'
        if group and audit['branch'] != 'frag_gap':
            candidates, ng = self.group_rescore(mid, candidates, before, scores, ice, gl)
            audit['group_fragment_groups'] = ng
            if ng:
                audit['branch'] += '+group_fragment'
        result = scope['safe_extend'](candidates[:25], candidates[25:] + original) if candidates else original
        if audit['branch'] == 'ice_fallback':
            without_ice = scope['safe_extend'](before[:25], before[25:] + original) if before else original
            audit['fallback_changed'] = result != without_ice
            audit['fallback_changed_top1'] = result[:1] != without_ice[:1]
        return (result, audit)
