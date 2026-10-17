'''OOP orchestration
Keep the sergiomalv BIO entries first and add old prvsiyan BIO structures absent by engine2 source key. Recalculate added fingerprints and masses with engine2 native fp_and_mass and the same selected bits. No assumptions about old fingerprint-bit ordering. Source-key overlaps keep the new version; full official metric deduplication is NOT introduced into the pool. Candidate discovery is pinned to sergiomalv; the old dataset is read only by the extension. A 1200s preparation deadline raises rather than silently accepting partial coverage. More candidates may reduce accuracy or exceed runtime.'''

from pathlib import Path
import glob
import hashlib
import json
import time
import pandas as pd

class V22Stage:
    def __init__(self,label,indices):
        self.label,self.indices = label,indices

    def run(self,scope,sources):
        started = time.monotonic()
        print('STAGE:',self.label,flush=True)
        for index in self.indices:
            exec(compile(sources[index],f'<credited-v22-cell-{index}>','exec'),scope)
        return dict(stage=self.label,seconds=round(time.monotonic()-started,2))

class MetricDistinctRunnerUp:
    def __init__(self, native_row):
        self.native_row = native_row
        self.considered = self.alternatives = 0

    def rank(self,tokens,engine,key,limit=25,placeholder='CCO'):
        original = self.native_row(tokens,engine,key,limit,placeholder)
        if original[1] != 'same_as_rank1':
            return original
        self.considered += 1
        engine = [engine] if isinstance(engine,str) else list(engine or [])
        top_key = key(tokens[0])
        # Inspect physical top3 only; metric twins do not consume the alternative hedge slot.
        for smile in engine[:3]:
            if not isinstance(smile,str) or any(c in smile for c in ';\r\n'):
                continue
            candidate = smile.strip()
            identity = key(candidate)
            if identity is not None and identity != top_key:
                self.alternatives += 1
                return self.native_row(tokens,[candidate],key,limit,placeholder)
        return original

class V22Experiment:
    def __init__(self,sources,runner_up=False):
        self.sources,self.runner_up = sources,runner_up
        self.scope = {'__name__':'__main__'}

    @staticmethod
    def check_bio(paths):
        assert paths, 'Missing sergiomalv/casmi-chebi-lmsd-structures (bio_meta.pkl)'
        assert all('casmi-chebi-lmsd-structures' in Path(p).parts for p in paths), (
            'Ambiguous BIO inputs: remove old prvsiyan/chebi-lipidmaps-casmi26; retain the V22 sergiomalv dataset',paths)

    def run(self):
        bio=glob.glob('/kaggle/input/**/bio_meta.pkl',recursive=True)
        assert any('casmi-chebi-lmsd-structures' in Path(p).parts for p in bio), 'Attach sergiomalv BIO'
        assert any('chebi-lipidmaps-casmi26' in Path(p).parts for p in bio), 'Attach prvsiyan BIO for union'
        assert all(any(s in Path(p).parts for s in ('casmi-chebi-lmsd-structures','chebi-lipidmaps-casmi26')) for p in bio), ('Unexpected BIO input',bio)
        scope = self.scope
        try:
            from IPython.display import display
        except ImportError:
            display = print
        scope['display'] = display
        stages = [V22Stage('Runtime, verified inputs and smoke switch',[3,5,6,8,9,11]),
            V22Stage('Engine2 and PubChem retrieval',[13,15]),
            V22Stage('Main engine and simulator evidence',[17,18,20,22]),
            V22Stage('rel150 promotion, fusion and official metric audit',[24,26,28,29,30,31]),
            V22Stage('Source fragment-gap, E006 and metric-exact rank2',[33,34,36])]
        timing = [stage.run(scope,self.sources) for stage in stages]
        assert scope['E2R2_STATS'] is not None, 'Source E2R2 did not complete'
        status = json.loads(Path('e2r2_stats.json').read_text())
        assert status.get('status')=='applied', 'Source E2R2 failed; do not submit a silently degraded recipe'
        control_bytes = Path('submission.csv').read_bytes()
        Path('submission_v22_control.csv').write_bytes(control_bytes)
        control = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        details = {}
        if self.runner_up:
            wrapper = MetricDistinctRunnerUp(scope['e2r2_row'])
            scope['e2r2_row'] = wrapper.rank
            workdir = Path('runner_up_work')
            workdir.mkdir(exist_ok=True)
            pre_hedge = Path('submission_base_e2r2.csv').read_bytes()
            (workdir/'submission.csv').write_bytes(pre_hedge)
            scope['_e2r2_apply'](scope,pre_hedge,workdir=str(workdir))
            treatment_bytes = (workdir/'submission.csv').read_bytes()
            treatment = pd.read_csv(workdir/'submission.csv',dtype=str,keep_default_na=False)
            assert treatment.columns.tolist() == control.columns.tolist()
            assert treatment.molecule_id.tolist() == control.molecule_id.tolist()
            assert (treatment.smiles.str.split(';').str[0] == control.smiles.str.split(';').str[0]).all()
            Path('submission.csv').write_bytes(treatment_bytes)
            details = dict(agreement_rows=wrapper.considered,runner_up_candidates=wrapper.alternatives)
        final = pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        comparison = pd.DataFrame(dict(molecule_id=control.molecule_id,changed=final.smiles!=control.smiles))
        comparison.to_csv('v22_comparison.csv',index=False)
        report = dict(mode='bio_union',
            measured_score=None,source_score_user_reported=.451,changed_rows_vs_same_run_source_output=int(comparison.changed.sum()),
            control_sha256=hashlib.sha256(control_bytes).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),details=details,stages=timing)
        Path('v22_adaptation_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        if self.runner_up and not report['changed_rows_vs_same_run_source_output']:
            print('NO-OP versus same-run V22: identical predictions do not merit a separate submission.')
        return report
