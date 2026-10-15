'''Iteration 07 — Replicated Fragment Evidence
An independent treatment on Initial01 (user-confirmed public 0.441), not an overlay on submitted Iteration06 predictions. Hypothesis: a single unusually favourable spectrum can dominate mean contrast evidence. Replace its aggregation with a per-candidate median and require at least three distinct informative spectra. Exact duplicates of the prepared normalized intensities and neutral mass queries count once; distinct spectra are not guaranteed statistically independent. Each informative spectrum must have at least two discriminative peaks carrying 5% intensity. Use the same two-cleavage kernel, 2..8 candidate groups, 40-bond cap, no-forward-evidence molecule gate, 900s hidden / 120s smoke budget and aggressive lambda 1.25 as Iteration06. The method tests persistent candidate support, not exact repeated fragment identities. Groups with fewer than three informative spectra retain their original 0.441 order. Risk: collision energies can legitimately favour different structures; median filtering can discard useful evidence. No score improvement is claimed. All upstream credits and links remain. No new datasets: retain every input of the successful baseline, including the compatible RDKit2025 simulator wheel and separate RDKit2026 main-engine wheel. Check iteration_audit.json for actual changed rows; identical outputs must not consume a submission.'''
class EvidenceIteration:
    def __init__(self, experiment):
        self.experiment, self.scope = experiment, experiment.scope
        self.audit = []

    @staticmethod
    def finite(value):
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def value(self, evidence, smile, key):
        value = evidence.get(smile)
        return evidence.get(key) if value is None else value

    def fused(self, mid):
        scope = self.scope
        engine = scope['ENG'].get(mid)
        if not engine or not engine.get('smiles') or mid not in scope['PREFUSION_LISTS']:
            return [], [], [], []
        first = [s for s in scope['PREFUSION_LISTS'][mid] if s and s != 'CCO']
        candidates, scores, _ = MetricIdentityFusion(scope).rank(first, engine['smiles'], 40)
        keys = [scope['chem'].score_key(s) or s for s in candidates]
        formulas = [scope['_form2'](s) for s in candidates]
        return candidates, scores, keys, formulas

    def baseline_order(self, mid, candidates, scores, keys, formulas):
        scope = self.scope
        ice, gl = scope['ICE_SCORES'].get(mid, {}), scope['gl_of'](mid)
        if ice and gl:
            return scope['gl_fuse'].rerank_multi(candidates, keys, scores, formulas,
                [{}, gl], [scope['ICE_LAM'], scope['GL_LAM']], top_n=len(candidates))
        frag = scope.get('FRAG_GAP', {}).get(mid)
        if not gl and frag and not any(v is not None for v in (ice or {}).values()):
            return scope['gl_fuse'].rerank_multi(candidates, keys, scores, formulas,
                [frag], [.5], top_n=len(candidates))
        if ice and not gl:
            return scope['ice_fuse'].rerank(candidates, keys, scores, formulas, ice,
                lam=scope['ICE_LAM'], top_n=len(candidates))
        return list(range(len(candidates)))

    def preserve_slots(self, order, eligible):
        assert sorted(order) == list(range(len(order)))
        assert all(i == j for i,j in enumerate(order) if i not in eligible)

    def finish(self, rows, control, name, details):
        frame = pd.DataFrame(rows, columns=['molecule_id','smiles'])
        active = set(self.scope['te'].molecule_id.astype(str))
        SubmissionBuilder(self.scope).validate(frame, control, active)
        frame.to_csv('submission.csv', index=False)
        comparison = pd.DataFrame(dict(molecule_id=control.molecule_id,
            changed=control.smiles != frame.smiles,
            changed_top1=control.smiles.str.split(';').str[0] != frame.smiles.str.split(';').str[0]))
        comparison.to_csv('iteration_vs_0441.csv', index=False)
        pd.DataFrame(self.audit).to_csv('iteration_group_audit.csv', index=False)
        report = dict(experiment=name, changed_rows=int(comparison.changed.sum()),
            changed_top1=int(comparison.changed_top1.sum()), details=details, measured_score=None,
            control_sha256=hashlib.sha256(Path('submission_0441_control.csv').read_bytes()).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest())
        Path('iteration_audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not report['changed_rows']:
            print('NO-OP versus Initial 01: do not spend a second submission on identical predictions.')
        return report

    def control(self):
        Path('submission_0441_control.csv').write_bytes(Path('submission.csv').read_bytes())
        return pd.read_csv('submission.csv', dtype=str, keep_default_na=False)

class ReplicatedFragmentFallback(EvidenceIteration):
    @staticmethod
    def contrast(credits, intensity):
        # A shared explanation carries no information about which candidate is correct.
        support = (credits > 0).sum(axis=0)
        n = credits.shape[0]
        information = np.log(n / np.maximum(support,1)) / np.log(n)
        informative = (support > 0) & (support < n)
        if informative.sum() < 2 or float(intensity[informative].sum()) < .05:
            return None
        return (credits * information[None,:]) @ intensity

    def score(self, spectra, candidates, deadline):
        module = self.scope.get('frag_rescore')
        if module is None or time.monotonic() >= deadline:
            return {}
        try:
            for smile in candidates:
                graph = module.mol_graph(smile)
                if graph is None or len(graph[1]) > 40:
                    return {}
            cfg = module.resolve_cfg(dict(max_breaks=2))
            levels = [module.candidate_levels(s,cfg,self.cache) for s in candidates]
            if any(level is None for level in levels):
                return {}
            scores, seen = [], set()
            for spectrum in spectra:
                prep = module._prep_spectrum(spectrum,cfg)
                if prep is None:
                    continue
                iw,q,ns,npk = prep
                signature = (iw.tobytes(), q.tobytes(), ns, npk)
                if signature in seen:
                    continue
                seen.add(signature)
                credits = np.zeros((len(candidates),npk))
                for i,level in enumerate(levels):
                    for masses,weight in zip(level,(1.,cfg['w1'],cfg['w2'])):
                        if len(masses):
                            hits = (np.searchsorted(masses,q+cfg['tol'],'right') >
                                np.searchsorted(masses,q-cfg['tol'],'left')).reshape(ns,npk).any(axis=0)
                            credits[i] = np.maximum(credits[i],hits*weight)
                result = self.contrast(credits,iw)
                if result is not None:
                    scores.append(result)
            if len(scores) < 3:
                return {}
            values = np.median(scores,axis=0)
            if not np.isfinite(values).all() or np.ptp(values) <= 1e-8:
                return {}
            return dict(zip(candidates,map(float,values)))
        except Exception as error:
            self.errors.append(repr(error))
            return {}

    def run(self):
        scope, control = self.scope, self.control()
        self.cache, self.errors, self.audit = {}, [], []
        deadline = time.monotonic() + (900 if scope['IS_RERUN'] else 120)
        frames = {str(mid):frame for mid,frame in scope['te'].groupby('molecule_id',sort=False)}
        rows, accepted = [], 0
        for row in control.itertuples(index=False):
            mid,original = row.molecule_id,row.smiles.split(';')
            revised = list(original)
            ice,gl = scope['ICE_SCORES'].get(mid,{}),scope['gl_of'](mid)
            if mid in frames and not gl and not any(v is not None for v in ice.values()):
                c,s,k,f = self.fused(mid)
                baseline = self.baseline_order(mid,c,s,k,f)
                current = [c[i] for i in baseline]
                assert scope['safe_extend'](current[:25],current[25:]+original) == original
                spectra = [dict(mz=np.asarray(r.ms2_mzs),it=np.asarray(r.ms2_normalized_intensities),
                    adduct=r.adduct,prec=float(r.precursor_mz),mode=1 if r.ionization_mode=='positive' else -1)
                    for r in frames[mid].itertuples(index=False)][:16]
                groups = {}
                for i,formula in enumerate(f):
                    groups.setdefault(formula,[]).append(i)
                evidence,eligible = {},set()
                positions = {i:j for j,i in enumerate(baseline)}
                for formula,indices in groups.items():
                    if not 2 <= len(indices) <= 8:
                        continue
                    complete = self.score(spectra,[c[i] for i in indices],deadline)
                    self.audit.append(dict(molecule_id=mid,formula=formula,candidates=len(indices),accepted=bool(complete)))
                    if complete:
                        evidence.update(complete)
                        eligible.update(positions[i] for i in indices)
                        accepted += 1
                if evidence:
                    by_smile = {smile:i for i,smile in enumerate(c)}
                    aligned = [by_smile[smile] for smile in current]
                    order = scope['gl_fuse'].rerank_multi(current,[k[i] for i in aligned],
                        [s[i] for i in aligned],[f[i] for i in aligned],[evidence],[1.25],top_n=len(current))
                    self.preserve_slots(order,eligible)
                    current = [current[i] for i in order]
                    revised = scope['safe_extend'](current[:25],current[25:]+original)
            rows.append((mid,';'.join(revised)))
        return self.finish(rows,control,'replicated_fragment_fallback',dict(target_groups=len(self.audit),
            accepted_groups=accepted,errors=self.errors,max_group_size=8,max_bonds=40,contrast_lambda=1.25,min_distinct_informative_spectra=3,aggregation="median",
            budget_seconds=900 if scope['IS_RERUN'] else 120))
iteration = ReplicatedFragmentFallback(experiment)
iteration_result = iteration.run()
