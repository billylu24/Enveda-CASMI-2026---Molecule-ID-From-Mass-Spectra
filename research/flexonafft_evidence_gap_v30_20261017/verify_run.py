"""Verify exact remote source, runtime acceptance, BIO counts and prediction parity."""
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import pandas as pd
from prepare_release import ROOT,SOURCE,KERNEL,VERSION,code

EVID=ROOT/'evidence'
def load(path):return json.loads(path.read_text())
def main():
    dest=EVID/'platform';ref=EVID/'source_reference'
    state=load(EVID/'publication_state.json');remote=load(dest/'api_metadata.json')
    version=state['push_response']['versionNumber']
    assert remote['currentVersionNumber']==version
    assert load(dest/'kernel_status.json')['status']=='COMPLETE'
    assert code(load(dest/'platform_source.ipynb'))==code(load(ROOT/'release/reproduction.ipynb'))
    assert code(load(ref/'platform_source.ipynb'))==code(load(ROOT/'source/casmi26-closing-the-evidence-gap.ipynb'))
    accept=load(dest/'reproduction_acceptance.json');assert accept['passed'] and accept['source_version']==VERSION
    active=set(accept['processed_ids']);assert active==set(load(ref/'e004_verification.json')['processed_ids'])
    run=pd.read_csv(dest/'submission.csv',dtype=str,keep_default_na=False).set_index('molecule_id')
    upstream=pd.read_csv(ref/'submission.csv',dtype=str,keep_default_na=False).set_index('molecule_id')
    assert set(run.index)==set(upstream.index) and len(run)==accept['rows']
    for name,frame in [('platform',run),('source_reference',upstream)]:
        frame.loc[sorted(active)].reset_index().to_csv(EVID/name/'processed_predictions.csv',index=False)
    comparison=[]
    for mid in sorted(active):
        a,b=upstream.loc[mid,'smiles'].split(';'),run.loc[mid,'smiles'].split(';')
        comparison.append(dict(molecule_id=mid,top1_equal=a[0]==b[0],candidate_sets_equal=set(a)==set(b),full_ranking_equal=a==b,
            differences=[dict(rank=i+1,upstream=x,reproduction=y) for i,(x,y) in enumerate(zip_longest(a,b)) if x!=y],
            upstream_candidates=len(a),reproduction_candidates=len(b)))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(dest/'submission.csv')==accept['submission_sha256']
    ubio=load(ref/'eng/bio_union_audit.json');bio=accept['bio_union']
    count_fields=['new_rows','old_rows','added_rows','overlap_rows','invalid_rows']
    bio_exact=all(ubio[k]==bio[k] for k in count_fields)
    assert bio_exact
    for simulator,meta in accept['simulators'].items():
        assert meta['status']=='ok' and meta['device'].startswith('cuda') and meta['rdkit']=='2025.03.6'
        assert not meta.get('errors') and meta.get('n_pred_fail',0)==0 and meta.get('n_feat_fail',0)==0
    inputs=load(dest/'e004_candidate_inputs.json'); source_inputs=load(ref/'e004_candidate_inputs.json')
    input_agreement={name:dict(matching_molecules=sum(values.get(mid)==inputs.get(name,{}).get(mid) for mid in active),total=len(active)) for name,values in source_inputs.items()}
    report=dict(candidate_input_comparison=input_agreement,source=SOURCE,source_version=VERSION,kernel=KERNEL,version=version,execution_acceptance_passed=True,
        original_code_exact=True,processed=len(active),rows=len(run),placeholders=accept['placeholders'],
        top1_matches=sum(r['top1_equal'] for r in comparison),candidate_set_matches=sum(r['candidate_sets_equal'] for r in comparison),
        full_ranking_matches=sum(r['full_ranking_equal'] for r in comparison),
        full_output_parity_passed=sha(dest/'submission.csv')==sha(ref/'submission.csv'),
        source_submission_sha256=sha(ref/'submission.csv'),reproduction_submission_sha256=sha(dest/'submission.csv'),
        bio_counts_exact=bio_exact,bio_counts={k:bio[k] for k in count_fields},runtime_seconds=accept['runtime_seconds'],
        measured_score=None,comparison=comparison)
    (EVID/'platform_acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='comparison'},indent=2))
    return report

if __name__=='__main__':main()
