import json
import sys
import types

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from baseline import PROTON, vectorize
from casmi_ml import adversarial_release as release
from casmi_ml.data import fingerprint


def raw_key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]


def mass(smiles):
    return rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smiles))


def spectrum(smiles, peaks, *, name='query'):
    return {'molecule_id':name,'inchikey14':raw_key(smiles),'canonical_smiles':smiles,
            'identity':release.identity(smiles),'precursor_mz':mass(smiles)+PROTON,
            'adduct':'[M+H]+','ionization_mode':'positive','ms2_mzs':peaks,
            'ms2_normalized_intensities':[1.] * len(peaks)}


def pool_row(smiles, origin='public'):
    return {'inchikey14':raw_key(smiles),'normalized_smiles':smiles,'mass':mass(smiles),
            'identity':release.identity(smiles),'formal_charge':0,'origin':origin}


def test_neural_candidate_fingerprints_match_canonical_training_graph():
    enol, keto = 'CC(O)=C', 'CC(=O)C'
    assert release.identity(enol) == release.identity(keto)
    assert not np.array_equal(fingerprint(enol),fingerprint(keto))
    assert np.array_equal(release.canonical_fingerprint(enol),fingerprint(keto))
    assert np.array_equal(release.canonical_fingerprint(enol),release.canonical_fingerprint(keto))


def test_unknown_reference_mask_checks_actual_alias_graph_and_raw_key():
    enol, keto, other = 'CC(O)=C','CC(=O)C','CCC=O'
    records = [(mass(s),key,s,{500:1.}) for key,s in
               [('unlisted_alias',enol),('another_alias',keto),('blocked_raw',other),('allowed',other)]]
    result = release.mask_unknown_records(records,{release.identity(keto)},{'blocked_raw'})
    assert [r[1] for r in result] == ['allowed']


def test_known_reference_mask_removes_all_binned_query_copies_across_sources():
    query = pd.DataFrame([spectrum('CC(=O)C',[40.01,50.01])])
    # Same binned representation despite different raw exact peaks and labels.
    copy = vectorize([40.02,50.02],[1.,1.])
    # Vector weights retain exact m/z, so a distinct exact peak vector is not a copy.
    assert copy != vectorize([40.01,50.01],[1.,1.])
    vector = vectorize([40.01,50.01],[1.,1.])
    records = [(58.,'query','CC(=O)C',vector),
               (58.,'alias_copy','CC(O)=C',dict(reversed(list(vector.items())))),
               (58.,'other_view','CC(=O)C',vectorize([42.,51.],[1.,.5]))]
    result = release.mask_query_copies(records,query)
    assert [r[1] for r in result] == ['other_view']


def test_canonical_blend_recovers_slots_after_alias_deduplication():
    library = [('enol','CC(O)=C',.9),('keto','CC(=O)C',.89)]
    library += [(f'alkane_{n}','C'*n,.8-n*.01) for n in range(1,31)]
    result = release.canonical_blend(library,[])
    assert len(result) == 25
    assert result[0] == ('enol','CC(O)=C')
    assert len({release.identity(smi) for _,smi in result}) == 25
    assert ('alkane_24','C'*24) in result


def test_unique_pairs_preserves_first_official_identity_and_limit():
    result = release.unique_pairs([('enol','CC(O)=C'),('keto','CC(=O)C'),('other','CCC=O')],limit=2)
    assert result == [('enol','CC(O)=C'),('other','CCC=O')]
    assert release.reciprocal_rank(release.identity('CC(=O)C'),result) == 1.


def test_ranker_shares_pool_and_alias_representatives_between_neural_controls():
    enol,keto,other = 'CC(O)=C','CC(=O)C','CCC=O'
    records = [(mass(enol),raw_key(enol),enol,vectorize([40.,44.],[1.,.8]))]
    coconut = pd.DataFrame([{'inchikey':raw_key(keto),'canonical_smiles':keto,'exact_mass':mass(keto)}])
    pool = pd.DataFrame([pool_row(enol),pool_row(keto),pool_row(other)]).sort_values(['mass','inchikey14']).reset_index(drop=True)
    engine = release.Ranker(records,pool,coconut,())
    query = pd.DataFrame([spectrum(keto,[51.])])
    probability = release.canonical_fingerprint(keto).astype(float)
    result,audit,candidates = engine.rank(query,{'baseline':probability,'cgan':probability})
    assert result['baseline_raw'] == result['cgan_raw']
    assert result['baseline_routed'] == result['cgan_routed']
    assert len(candidates) == 2
    assert audit['canonical_duplicates_removed'] == 1
    for arm,pairs in result.items():
        assert len({release.identity(s) for _,s in pairs}) == len(pairs),arm
    assert audit['_available_identities']['baseline_raw'] == audit['_available_identities']['cgan_raw']
    assert result['baseline_raw'][0][0] == raw_key(enol)


def test_metrics_use_each_arms_available_candidate_pool():
    rows = [{'covered':False,'covered_by_arm':{'historical':True,'cgan_raw':False},
             'rr':{'historical':.5,'cgan_raw':0.}}]
    historical = release.metrics(rows,'historical')
    neural = release.metrics(rows,'cgan_raw')
    assert historical['candidate_recall'] == 1.
    assert historical['candidate_conditional_mrr'] == .5
    assert neural['candidate_recall'] == 0.
    assert neural['candidate_conditional_mrr'] is None


def test_empty_known_subset_returns_json_serializable_null_metrics():
    report,rows = release.evaluate(pd.DataFrame(),None,{'baseline':None,'cgan':None})
    assert rows == []
    assert report['metrics']['cgan_routed']['molecules'] == 0
    assert report['cgan_vs_historical']['difference'] is None
    json.dumps(report,allow_nan=False)


def test_pairwise_difference_is_molecule_paired_and_deterministic():
    rows = [{'rr':{'gan':.5,'control':1.}},{'rr':{'gan':1.,'control':0.}}]
    result = release.difference(rows,'gan','control',samples=200)
    assert result['difference'] == .25
    assert result == release.difference(rows,'gan','control',samples=200)
    assert result['bootstrap_unit'] == 'official canonical molecule'


def test_graph_edit_pool_is_shared_and_separates_new_exact_coverage(monkeypatch):
    from casmi_ml import adversarial
    anchor,novel = 'CCC(=O)CC','CCCCC=O'
    assert mass(anchor) == mass(novel)
    records = [(mass(anchor),raw_key(anchor),anchor,vectorize([42.],[1.]))]
    coconut = pd.DataFrame([{'inchikey':raw_key(anchor),'canonical_smiles':anchor,'exact_mass':mass(anchor)}])
    pool = pd.DataFrame([pool_row(anchor)])
    engine = release.Ranker(records,pool,coconut,())
    query = pd.DataFrame([spectrum(novel,[51.])])
    calls = []
    module = types.ModuleType('casmi_ml.graph_candidates')

    def generate(anchors,excluded_identities,seed):
        calls.append((anchors,excluded_identities,seed))
        return [pool_row(novel)],{'generated':1,'method':'controlled test edit'}

    module.generate_candidates = generate
    monkeypatch.setitem(sys.modules,'casmi_ml.graph_candidates',module)
    probability = release.canonical_fingerprint(novel).astype(float)
    result,audit,candidates = engine.rank(query,{'baseline':probability,'cgan':probability})
    assert calls[0][0] == [anchor]
    assert release.identity(anchor) in calls[0][1]
    assert result['baseline_raw'] == result['cgan_raw']
    assert result['baseline_routed'] == result['cgan_routed']
    assert release.identity(novel) not in {release.identity(s) for _,s in result['expanded_chemistry']}
    assert audit['newgraph_count'] == 1
    assert audit['graph_output_count']['cgan_raw'] == 1
    assert len(candidates) == 2
    monkeypatch.setattr(adversarial,'predict_fingerprints',lambda *args,**kwargs:np.asarray([probability]))
    report,rows = release.evaluate(query,engine,{'baseline':('x',{'preprocessing':{}}),'cgan':('x',{'preprocessing':{}})})
    assert report['metrics']['expanded_chemistry']['candidate_recall'] == 0.
    assert report['metrics']['cgan_raw']['candidate_recall'] == 1.
    assert report['graph_only_newcoverage_molecules'] == 1
    assert rows[0]['original_pool_covered'] is False
    assert rows[0]['graph_only_newcoverage'] is True


def test_query_graph_seed_ignores_truth_and_identity_labels():
    query = pd.DataFrame([spectrum('CCCCC=O',[51.])])
    relabeled = query.assign(inchikey14='OTHER_RAW_LABEL',identity='OTHER_IDENTITY',canonical_smiles='CCO',molecule_id='other')
    assert release.query_seed(query) == release.query_seed(relabeled)


def test_confident_protection_bypasses_graph_generation(monkeypatch):
    anchor = 'CCC(=O)CC'
    records = [(mass(anchor),raw_key(anchor),anchor,vectorize([42.],[1.]))]
    coconut = pd.DataFrame([{'inchikey':raw_key(anchor),'canonical_smiles':anchor,'exact_mass':mass(anchor)}])
    engine = release.Ranker(records,pd.DataFrame([pool_row(anchor)]),coconut,())
    query = pd.DataFrame([spectrum(anchor,[42.])])
    module = types.ModuleType('casmi_ml.graph_candidates')
    module.generate_candidates = lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('guard should bypass graph edits'))
    monkeypatch.setitem(sys.modules,'casmi_ml.graph_candidates',module)
    result,audit,_ = engine.rank(query,{'cgan':release.canonical_fingerprint(anchor).astype(float)})
    assert audit['protected']
    assert audit['newgraph_count'] == 0
    assert result['cgan_routed'] == result['historical']


def test_full_run_keeps_known_remaining_references_without_oracle_injection(tmp_path,monkeypatch):
    from casmi_ml import adversarial
    data = tmp_path/'data'
    data.mkdir()
    work = tmp_path/'work'
    train_smi,dev_smi,accept_smi = 'CCO','CCC=O','CC(=O)C'
    train = pd.DataFrame([spectrum(train_smi,[42.])])
    dev = pd.DataFrame([spectrum(dev_smi,[45.])])
    accept = pd.DataFrame([spectrum(accept_smi,[40.,50.])])
    pd.DataFrame([spectrum(accept_smi,[40.,50.])]).to_parquet(data/'test.parquet')
    catalog = pd.DataFrame([pool_row(s,'library') for s in [train_smi,dev_smi,accept_smi,'CC(O)=C']])
    coconut = pd.DataFrame([{'inchikey':raw_key(train_smi),'canonical_smiles':train_smi,'exact_mass':mass(train_smi)}])
    coconut.to_parquet(tmp_path/'coconut.parquet')
    extra = pd.DataFrame(columns=['inchikey14','normalized_smiles','mass','formal_charge'])
    extra.to_parquet(tmp_path/'extra.parquet')
    (tmp_path/'rules.json').write_text(json.dumps({'version':1,'rules':[]}))

    def prepare(_train_path,root,**kwargs):
        root.mkdir()
        catalog.to_parquet(root/'catalog.parquet')
        for name,frame in [('train',train),('dev',dev),('acceptance',accept)]:
            frame.to_parquet(root/f'{name}.parquet')
        return {'test':'controlled fixture'}

    monkeypatch.setattr(adversarial,'prepare',prepare)
    monkeypatch.setattr(adversarial,'train_pair',lambda *args,**kwargs:{'test':'fixed weights'})
    monkeypatch.setattr(adversarial,'load_generator',lambda path:('model',{'preprocessing':{}}))
    monkeypatch.setattr(adversarial,'predict_fingerprints',lambda model,frame,prep,**kwargs:
                        np.asarray([release.canonical_fingerprint(s) for s in frame.canonical_smiles],dtype=np.float32))
    vector = vectorize([40.,50.],[1.,1.])
    records = [(mass(accept_smi),raw_key(accept_smi),accept_smi,vector),
               (mass(accept_smi),'alias_not_in_catalog','CC(O)=C',vector),
               (mass(accept_smi),raw_key(accept_smi),accept_smi,vectorize([41.,51.],[1.,.8])),
               (mass(dev_smi),raw_key(dev_smi),dev_smi,vectorize([46.],[1.])),
               (mass(train_smi),raw_key(train_smi),train_smi,vectorize([43.],[1.]))]
    monkeypatch.setattr(release,'load_candidates',lambda *args,**kwargs:(records,{}))
    submission,report = release.run(data,tmp_path/'coconut.parquet',tmp_path/'extra.parquet',tmp_path/'rules.json',work)
    assert report['acceptance']['metrics']['cgan_raw']['candidate_recall'] == 0.
    known = report['known_acceptance']
    assert known['protocol']['remaining_reference_spectra'] == 3
    assert known['protocol']['eligible_known_molecules'] == 1
    assert known['metrics']['cgan_raw']['candidate_recall'] == 1.
    assert known['protocol']['no_truth_candidates_injected']
    assert len(submission) == 1
    assert (work/'known_acceptance_cases.json').exists()
    assert (work/'prepared_manifest.json').exists()
    assert not (work/'prepared').exists()
    json.dumps(report,allow_nan=False)
