"""Frozen fingerprint GAN, shared graph edits, acceptance, offline inference.

The GAN produces fingerprints, not molecular graphs. Bounded graph edits of
observed retrieval anchors augment the independently supplied structure pool.
"""
import gc
import hashlib
import json
import shutil
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from baseline import ADDUCT_MASS, load_candidates, make_matrix, vectorize
from hybrid import blend, coconut_rank, library_rank
from casmi_ml.chemical_priors import candidate_scores, extract_evidence, load_rules, rerank
from casmi_ml.data import fingerprint, write_json
from casmi_ml.hybrid_chemistry import COCONUT_COLUMNS, prepare_analog_pool

GUARD = .75
NEURAL_WEIGHT = .35
CHEMICAL_WEIGHT = .1
ENUMERATOR = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=500000)
def identity(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not mol.GetNumAtoms():
        return None
    return Chem.MolToInchiKey(ENUMERATOR.Canonicalize(mol))[:14]


def canonical_fingerprint(smiles):
    """Use the same tautomer representation as the prepared neural targets."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not mol.GetNumAtoms():
        return None
    return fingerprint(Chem.MolToSmiles(ENUMERATOR.Canonicalize(mol)))


def canonical_blend(library, analog):
    """Keep legacy source scores/mixing while sharing official identity slots.

    Deduplication must happen before the legacy 25-slot limit, so aliases do not
    prevent later distinct structures from filling the output.
    """
    representatives = {}
    sources = []
    for source in (library, analog):
        kept, seen = [], set()
        for key, smiles, score in source:
            official = identity(smiles)
            if not official or official in seen:
                continue
            seen.add(official)
            representatives.setdefault(official, (key, smiles))
            rep_key, rep_smiles = representatives[official]
            kept.append((rep_key, rep_smiles, score))
        sources.append(kept)
    return blend(*sources)


def mask_unknown_records(records, excluded_identities, excluded_raw=()):
    """Mask raw aliases and verify actual reference graphs independently."""
    excluded_identities, excluded_raw = set(excluded_identities), set(excluded_raw)
    return [r for r in records if r[1] not in excluded_raw
            and identity(r[2]) not in excluded_identities]


def vector_signature(vector):
    return tuple(sorted((int(k), float(v)) for k, v in vector.items()))


def query_seed(group):
    """Derive edit randomness solely from observed spectra, never query labels."""
    def finite(value):
        try:
            number=float(value)
        except (TypeError,ValueError):
            return None
        return number if np.isfinite(number) else None
    spectra = []
    for row in group.itertuples():
        spectra.append(json.dumps([str(row.adduct),str(row.ionization_mode),finite(row.precursor_mz),
                                   [finite(v) for v in row.ms2_mzs],
                                   [finite(v) for v in row.ms2_normalized_intensities]],
                                  separators=(',',':'),allow_nan=False))
    digest = hashlib.sha256('\n'.join(sorted(spectra)).encode()).digest()
    return int.from_bytes(digest[:8],'big')


def mask_query_copies(records, queries):
    """Remove held-out queries and all copies under the retrieval representation.

    Copies are excluded across raw identities and sources, even when the same
    binned vector belongs to a different metadata label.
    """
    signatures = {vector_signature(vectorize(row.ms2_mzs, row.ms2_normalized_intensities))
                  for row in queries.itertuples()}
    return [r for r in records if vector_signature(r[3]) not in signatures]


def unique_pairs(pairs, limit=25):
    """Deduplicate by the competition's tautomer-canonical connectivity identity."""
    output, seen = [], set()
    for key, smiles in pairs:
        canonical = identity(smiles)
        if canonical and canonical not in seen:
            output.append((key, smiles))
            seen.add(canonical)
        if len(output) == limit:
            break
    return output


def reciprocal_rank(truth, pairs):
    keys = [identity(smi) for _, smi in unique_pairs(pairs)]
    return 1. / (keys.index(truth) + 1) if truth in keys else 0.


def fuse(base, neural, weight=NEURAL_WEIGHT):
    scores = {}
    for ranking, arm_weight in [(base, 1-weight), (neural, weight)]:
        for position, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.) + arm_weight / (60 + position)
    return sorted(scores, key=lambda key: (-scores[key], key))


def soft_tanimoto(probability, fps):
    probability = np.asarray(probability, np.float64)
    intersection = fps @ probability
    return intersection / np.maximum(fps.sum(1) + probability.sum() - intersection, 1e-9)


def build_pool(catalog, coconut, extra, excluded=()):
    library = catalog.loc[~catalog.identity.isin(set(excluded))].copy()
    library = library.loc[library.formal_charge.eq(0), ['inchikey14', 'normalized_smiles', 'mass']]
    library['origin'] = 'library'
    public, _, report = prepare_analog_pool(coconut, extra)
    public = public.rename(columns={'inchikey': 'inchikey14', 'canonical_smiles': 'normalized_smiles', 'exact_mass': 'mass'})
    public['inchikey14'] = public.inchikey14.str[:14]
    public['origin'] = 'public'
    pool = pd.concat([library, public[library.columns]], ignore_index=True)
    pool = pool.drop_duplicates('inchikey14').sort_values(['mass', 'inchikey14']).reset_index(drop=True)
    return pool, report


class Ranker:
    def __init__(self, records, pool, coconut, rules):
        self.records = records
        self.masses = np.asarray([r[0] for r in records])
        self.order = np.argsort(self.masses)
        self.matrix = make_matrix([r[3] for r in records])
        self.pool = pool
        self.pool_mass = pool.mass.to_numpy()
        self.coconut = coconut
        self.coco_order = np.argsort(coconut.exact_mass.to_numpy())
        self.coco_mass = coconut.exact_mass.to_numpy()[self.coco_order]
        self.rules = rules
        self.fp_cache = {}
        self.analog_fp_cache = {}
        self.old_fp_cache = {}

    def query(self, center):
        delta = max(center*35e-6, .006)
        return self.pool.iloc[np.searchsorted(self.pool_mass, center-delta):np.searchsorted(self.pool_mass, center+delta, side='right')].copy()

    def analog_scores(self, candidates, library, center):
        """Legacy graph fingerprints, also shared by both neural controls."""
        refs = [(fingerprint(smi),score) for _,smi,score in library[:25]]
        refs = [(fp,score) for fp,score in refs if fp is not None]
        scores = np.zeros(len(candidates))
        if refs and len(candidates):
            for smiles in candidates.normalized_smiles:
                if smiles not in self.analog_fp_cache:
                    self.analog_fp_cache[smiles] = fingerprint(smiles)
            fps = np.asarray([self.analog_fp_cache[s] for s in candidates.normalized_smiles],dtype=np.float32)
            ref_fps = np.asarray([fp for fp,_ in refs],dtype=np.float32)
            intersection = fps @ ref_fps.T
            union = fps.sum(1)[:,None] + ref_fps.sum(1)[None,:] - intersection
            scores = ((intersection / np.maximum(union,1)) * np.asarray([.5+.5*s for _,s in refs])[None,:]).max(1)
            ppm = abs(candidates.mass.to_numpy()-center)/max(center,1e-9)*1e6
            scores *= np.exp(-.5*(ppm/15)**2)
        return scores

    def rank(self, group, probabilities):
        center, library = library_rank(group, self.records, self.matrix, self.masses, self.order)
        analog = coconut_rank(center, library, self.coconut, self.coco_mass, self.coco_order, self.old_fp_cache)
        historical = canonical_blend(library, analog)
        confidence = library[0][2] if library else 0.
        candidates = self.query(center)
        # Candidate identity and fingerprint computation are query-local and cached;
        # no oracle/truth is added to this pool.
        values, ids = [], []
        for row in candidates.itertuples():
            if row.normalized_smiles not in self.fp_cache:
                self.fp_cache[row.normalized_smiles] = canonical_fingerprint(row.normalized_smiles)
            fp = self.fp_cache[row.normalized_smiles]
            if fp is not None and identity(row.normalized_smiles):
                ids.append(row.Index)
                values.append(fp)
        candidates = candidates.loc[ids].copy()
        candidates['identity'] = candidates.normalized_smiles.map(identity)
        candidates = candidates.drop_duplicates('identity')
        original_identities = set(candidates.identity)
        original_count = len(candidates)
        duplicates_removed = len(ids)-original_count
        fps = np.asarray([self.fp_cache[s] for s in candidates.normalized_smiles], dtype=np.float32).reshape(-1,2048)
        # All neural and common arms share one representative key per official
        # structure, including when a library raw key differs from a public alias.
        representatives = {identity(smi): key for key,smi in reversed(historical)}
        for row in candidates.itertuples():
            representatives.setdefault(row.identity, row.inchikey14)
        keys = [representatives[value] for value in candidates.identity]
        lookup = dict(zip(candidates.inchikey14, candidates.normalized_smiles))
        lookup.update(zip(keys, candidates.normalized_smiles))
        lookup.update(historical)
        evidence = extract_evidence(group, self.rules)  # audit all queries, including protected ones
        analog_score = self.analog_scores(candidates,library,center)
        analog_keys = [keys[i] for i in sorted(range(len(keys)), key=lambda i: (-analog_score[i], keys[i]))]
        base = [key for key,_ in historical]
        base_seen = set(base)
        base.extend(key for key in analog_keys if key not in base_seen)
        chemical, _ = candidate_scores(lookup, evidence, self.rules) if evidence else ({},{})
        common = rerank(base, chemical, CHEMICAL_WEIGHT) if evidence else base
        original_common_available = {identity(lookup[key]) for key in common}
        result = {'historical': historical,
                  'expanded_chemistry': unique_pairs([(key,lookup[key]) for key in common])}
        generated_identities, graph_report = set(), {'status':'guard_bypassed','generated':0}
        if confidence < GUARD:
            from casmi_ml.graph_candidates import generate_candidates
            delta = max(center*35e-6,.006)
            anchors = [smi for _,smi in historical
                       if abs(rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smi))-center) <= delta][:2]
            if not anchors:
                anchors = [lookup[key] for key in analog_keys[:2]]
            graph_rows,graph_report = generate_candidates(
                anchors,excluded_identities=original_identities|{identity(s) for _,s in historical},
                seed=query_seed(group))
            valid_graphs = []
            for row in graph_rows:
                if len(valid_graphs) >= 32:
                    break
                smiles = row['normalized_smiles']
                official = identity(smiles)
                fp = canonical_fingerprint(smiles)
                if (not official or fp is None or official in original_identities|generated_identities
                        or abs(float(row['mass'])-center)>delta):
                    continue
                generated_identities.add(official)
                valid_graphs.append({**row,'identity':official,'origin':'generated_graph_edit'})
                self.fp_cache[smiles] = fp
                representatives.setdefault(official,row['inchikey14'])
            if valid_graphs:
                candidates = pd.concat([candidates,pd.DataFrame(valid_graphs)],ignore_index=True)
                keys = [representatives[value] for value in candidates.identity]
                lookup.update(zip(keys,candidates.normalized_smiles))
                lookup.update(historical)
                fps = np.asarray([self.fp_cache[s] for s in candidates.normalized_smiles],dtype=np.float32).reshape(-1,2048)
                analog_score = self.analog_scores(candidates,library,center)
                analog_keys = [keys[i] for i in sorted(range(len(keys)),key=lambda i:(-analog_score[i],keys[i]))]
                base = [key for key,_ in historical]
                base_seen = set(base)
                base.extend(key for key in analog_keys if key not in base_seen)
                chemical,_ = candidate_scores(lookup,evidence,self.rules) if evidence else ({},{})
                common = rerank(base,chemical,CHEMICAL_WEIGHT) if evidence else base
        raw = {}
        for name,probability in probabilities.items():
            score = soft_tanimoto(probability,fps)
            raw[name] = [keys[i] for i in sorted(range(len(keys)),key=lambda i:(-score[i],keys[i]))]
        for name in probabilities:
            fused = fuse(common, raw[name])
            if confidence >= GUARD:
                chosen = [key for key,_ in historical]
            elif base:
                chosen = base[:1] + [key for key in fused if key != base[0]]
            else:
                chosen = fused
            result[name+'_raw'] = unique_pairs([(key,lookup[key]) for key in raw[name]])
            result[name+'_routed'] = unique_pairs([(key,lookup[key]) for key in chosen])
        audit = {'confidence': float(confidence), 'protected': confidence >= GUARD,
                 'pool_size': len(candidates), 'rule_matches': len(evidence),
                 'modes': sorted({str(v) for v in group.ionization_mode}),
                 'adducts': sorted({str(v) for v in group.adduct}),
                 'canonical_duplicates_removed':duplicates_removed,
                 'original_pool_size':original_count,'newgraph_count':len(generated_identities),
                 'graph_generation':graph_report,
                 'generated_candidates':[{'identity':row.identity,'smiles':row.normalized_smiles,
                                          'mass':float(row.mass)} for row in candidates.itertuples()
                                         if row.identity in generated_identities],
                 'graph_output_count':{name:sum(identity(s) in generated_identities for _,s in pairs)
                                       for name,pairs in result.items()},
                 'gan_changed_historical': result['cgan_routed'] != historical if 'cgan' in probabilities else False}
        historical_available = {identity(smi) for _,smi,_ in library+analog if identity(smi)}
        raw_available = set(candidates.identity)
        common_available = {identity(lookup[key]) for key in common}
        audit['_available_identities'] = {'historical':historical_available,
                                         'expanded_chemistry':original_common_available}
        audit['_original_pool_identities'] = original_identities
        audit['_generated_identities'] = generated_identities
        for name in probabilities:
            audit['_available_identities'][name+'_raw'] = raw_available
            audit['_available_identities'][name+'_routed'] = (historical_available if confidence >= GUARD
                                                              else common_available)
        return result, audit, candidates


def metrics(rows, name):
    values = np.asarray([row['rr'][name] for row in rows])
    covered = np.asarray([row.get('covered_by_arm',{}).get(name,row['covered']) for row in rows],dtype=bool)
    if not len(rows):
        return {'molecules':0,'candidate_recall':None,'mrr25':None,'top1':None,
                'top5':None,'top25':None,'candidate_conditional_mrr':None}
    return {'molecules': len(rows), 'candidate_recall': float(covered.mean()),
            'mrr25': float(values.mean()), 'top1': float((values==1).mean()),
            'top5': float((values>=.2).mean()), 'top25': float((values>0).mean()),
            'candidate_conditional_mrr': float(values[covered].mean()) if covered.any() else None}


def difference(rows, arm, control, samples=2000):
    values = np.asarray([r['rr'][arm]-r['rr'][control] for r in rows])
    if not len(values):
        return {'difference':None,'ci95':[None,None],'bootstrap_unit':'official canonical molecule',
                'samples':samples,'reason':'no eligible query molecules'}
    rng = np.random.default_rng(20261002)
    estimates = np.asarray([values[rng.integers(len(values),size=len(values))].mean() for _ in range(samples)])
    return {'difference': float(values.mean()), 'ci95': np.quantile(estimates,[.025,.975]).tolist(),
            'bootstrap_unit': 'official canonical molecule', 'samples': samples}


def evaluate(frame, engine, models, training_scaffolds=()):
    from casmi_ml.adversarial import predict_fingerprints
    from casmi_ml.failure_audit import rule_gate_audit
    if frame.empty:
        names = ['historical','expanded_chemistry'] + [name+suffix for name in models for suffix in ('_raw','_routed')]
        return {'metrics':{name:metrics([],name) for name in names},
                'cgan_vs_supervised_raw':difference([],'cgan_raw','baseline_raw'),
                'cgan_vs_supervised_routed':difference([],'cgan_routed','baseline_routed'),
                'cgan_vs_historical':difference([],'cgan_routed','historical'),
                'supervised_vs_expanded_chemistry':difference([],'baseline_routed','expanded_chemistry'),
                'cgan_vs_expanded_chemistry':difference([],'cgan_routed','expanded_chemistry'),
                'protected':0,'protected_top1_wrong':0,'rules_matched_queries':0,
                'canonical_duplicate_slots_removed':0,'strata':{},
                'reason':'no eligible query molecules'},[]
    outputs = {name: predict_fingerprints(model,frame,ckpt['preprocessing'],samples=8)
               for name,(model,ckpt) in models.items()}
    rows = []
    groups = frame.groupby('identity',sort=True).indices
    for number,(truth,indexes) in enumerate(groups.items(),1):
        group = frame.iloc[indexes]
        predictions = {name: probability[indexes].mean(0) for name,probability in outputs.items()}
        pairs,audit,candidates = engine.rank(group,predictions)
        available = audit.pop('_available_identities')
        original_pool = audit.pop('_original_pool_identities')
        generated = audit.pop('_generated_identities')
        mol = Chem.MolFromSmiles(group.iloc[0].canonical_smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        stratum = 'acyclic' if not scaffold else 'seen_scaffold' if scaffold in training_scaffolds else 'unseen_scaffold'
        row = {'identity': truth, 'raw_key': str(group.iloc[0].inchikey14),
               'truth_smiles': str(group.iloc[0].canonical_smiles),
               'covered': bool(truth in set(candidates.identity)),
               'covered_by_arm': {name:bool(truth in values) for name,values in available.items()},
               'original_pool_covered':bool(truth in original_pool),
               'graph_only_newcoverage':bool(truth in generated and truth not in original_pool),
               'candidate_count_by_arm': {name:len(values) for name,values in available.items()},
               'scaffold_stratum': stratum,
               'mass': float(rdMolDescriptors.CalcExactMolWt(mol)),
               'rr': {name: reciprocal_rank(truth,prediction) for name,prediction in pairs.items()},
               'top25': {name: [{'key':key,'smiles':smi,'identity':identity(smi)} for key,smi in prediction]
                         for name,prediction in pairs.items()}, **audit}
        row['chemical_audit'] = rule_gate_audit(group,engine.rules,row['truth_smiles'])
        rows.append(row)
        if number % 100 == 0:
            print(f'evaluated {number}/{len(groups)} canonical molecules',flush=True)
    names = list(rows[0]['rr'])
    report = {'metrics': {name: metrics(rows,name) for name in names},
              'cgan_vs_supervised_raw': difference(rows,'cgan_raw','baseline_raw'),
              'cgan_vs_supervised_routed': difference(rows,'cgan_routed','baseline_routed'),
              'cgan_vs_historical': difference(rows,'cgan_routed','historical'),
              'supervised_vs_expanded_chemistry': difference(rows,'baseline_routed','expanded_chemistry'),
              'cgan_vs_expanded_chemistry': difference(rows,'cgan_routed','expanded_chemistry'),
              'protected': sum(r['protected'] for r in rows),
              'protected_top1_wrong': sum(r['protected'] and r['rr']['historical']!=1 for r in rows),
              'rules_matched_queries': sum(r['rule_matches']>0 for r in rows),
              'canonical_duplicate_slots_removed': sum(r['canonical_duplicates_removed'] for r in rows),
              'generated_graph_candidates':sum(r['newgraph_count'] for r in rows),
              'queries_with_generated_graphs':sum(r['newgraph_count']>0 for r in rows),
              'graph_only_newcoverage_molecules':sum(r['graph_only_newcoverage'] for r in rows),
              'original_mass_pool_recall':float(np.mean([r['original_pool_covered'] for r in rows])),
              'graph_output_count':{name:sum(r['graph_output_count'][name] for r in rows) for name in names},
              'coverage_definition':'Per-arm actual available canonical candidate identities before Top25 truncation; no truth injection.',
              'chemical_gates':{'all_queries_evaluated_including_protected':True,
                               'truth_motif_supported_queries':sum(bool(r['chemical_audit']['truth_supported_rule_ids']) for r in rows),
                               'alias_restored_rule_matched_queries':sum(r['chemical_audit']['alias_restored_rule_matches']>0 for r in rows),
                               'encoding_issue_queries':sum(r['chemical_audit']['mode_or_whitespace_encoding_issue_observed'] for r in rows)},
              'strata': {stratum: {name: metrics([r for r in rows if r['scaffold_stratum']==stratum],name) for name in names}
                         for stratum in sorted({r['scaffold_stratum'] for r in rows})}}
    return report,rows


def run(data_dir,coconut_path,catalog_path,dictionary_path,working):
    """One frozen training experiment, two controls, one acceptance, GAN submission."""
    from casmi_ml.adversarial import prepare,train_pair,load_generator
    started=time.monotonic()
    working=Path(working)
    working.mkdir(parents=True,exist_ok=True)
    root=working/'prepared'
    manifest=prepare(Path(data_dir)/'train.parquet',root,test_path=Path(data_dir)/'test.parquet')
    comparison=train_pair(root,working/'models',seconds_per_arm=1800)
    models={name:load_generator(working/'models'/name/'model.pt') for name in ['baseline','cgan']}
    catalog=pd.read_parquet(root/'catalog.parquet')
    coconut=pd.read_parquet(coconut_path,columns=COCONUT_COLUMNS)
    extra=pd.read_parquet(catalog_path)
    rules=load_rules(dictionary_path)
    dev=pd.read_parquet(root/'dev.parquet')
    acceptance=pd.read_parquet(root/'acceptance.parquet')
    excluded=set(dev.identity)|set(acceptance.identity)
    excluded_raw=set(catalog.loc[catalog.identity.isin(excluded),'inchikey14'])
    evaluation=pd.concat([dev,acceptance],ignore_index=True)
    masses=evaluation.precursor_mz.to_numpy()-evaluation.adduct.map(ADDUCT_MASS).to_numpy()
    masses=masses[np.isfinite(masses)&(masses>0)]
    if not len(masses):
        raise ValueError('No supported finite precursor masses in evaluation cohort')
    all_records,_=load_candidates(Path(data_dir)/'train.parquet',masses)
    records=mask_unknown_records(all_records,excluded,excluded_raw)
    pool,pool_report=build_pool(catalog,coconut,extra,excluded)
    engine=Ranker(records,pool,coconut,rules)
    train=pd.read_parquet(root/'train.parquet',columns=['canonical_smiles'])
    scaffolds={MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in train.canonical_smiles.drop_duplicates()}
    dev_report,dev_rows=evaluate(dev,engine,models,scaffolds)
    acceptance_report,acceptance_rows=evaluate(acceptance,engine,models,scaffolds)
    write_json(working/'dev_ranking.json',dev_report)
    write_json(working/'acceptance_ranking.json',acceptance_report)
    write_json(working/'acceptance_cases.json',acceptance_rows)
    del engine,records
    gc.collect()
    # The known-spectrum condition uses exactly the same held-out query spectra
    # and frozen models. Only genuinely remaining references supply library
    # candidates; held-out labels are never appended to a candidate catalog.
    known_records=mask_query_copies(all_records,acceptance)
    observed_raw={r[1] for r in known_records}
    observed_identities={identity(r[2]) for r in known_records}
    known_catalog=catalog.loc[catalog.inchikey14.isin(observed_raw)
                              & catalog.identity.isin(observed_identities)]
    known_pool,_=build_pool(known_catalog,coconut,extra)
    known_queries=acceptance.loc[acceptance.identity.isin(observed_identities)].copy()
    known_engine=Ranker(known_records,known_pool,coconut,rules)
    known_report,known_rows=evaluate(known_queries,known_engine,models,scaffolds)
    known_report['protocol']={
        'query_cohort':'same acceptance molecules/spectra as unknown, subset with remaining same-identity references',
        'all_acceptance_query_vectors_and_binned_copies_removed':True,
        'remaining_reference_spectra':len(known_records),
        'acceptance_molecules_total':int(acceptance.identity.nunique()),
        'eligible_known_molecules':int(known_queries.identity.nunique()),
        'candidate_library':'actual remaining observed references plus independent public catalogs',
        'fit_and_routing_unchanged':True,
        'no_truth_candidates_injected':True,
    }
    write_json(working/'known_acceptance_ranking.json',known_report)
    write_json(working/'known_acceptance_cases.json',known_rows)
    generated_summary={'development':dev_report.get('generated_graph_candidates',0),
                       'unknown_acceptance':acceptance_report.get('generated_graph_candidates',0),
                       'known_acceptance':known_report.get('generated_graph_candidates',0)}
    del known_engine,known_records,all_records,known_pool,known_queries,known_rows
    del evaluation,dev,acceptance,train,dev_rows
    gc.collect()
    test=pd.read_parquet(Path(data_dir)/'test.parquet')
    masses=test.precursor_mz.to_numpy()-test.adduct.map(ADDUCT_MASS).to_numpy()
    records,_=load_candidates(Path(data_dir)/'train.parquet',masses[np.isfinite(masses)&(masses>0)])
    pool,_=build_pool(catalog,coconut,extra)
    engine=Ranker(records,pool,coconut,rules)
    from casmi_ml.adversarial import predict_fingerprints
    model,ckpt=models['cgan']
    probability=predict_fingerprints(model,test,ckpt['preprocessing'],samples=8)
    submissions,audits=[],[]
    for molecule_id,indexes in test.groupby('molecule_id',sort=False).indices.items():
        pairs,audit,_=engine.rank(test.iloc[indexes],{'cgan':probability[indexes].mean(0)})
        audit.pop('_available_identities')
        audit.pop('_original_pool_identities')
        audit.pop('_generated_identities')
        selected=pairs['cgan_routed']
        if not selected:
            raise ValueError(f'No GAN/historical candidates for {molecule_id}')
        submissions.append({'molecule_id':molecule_id,'smiles':';'.join(s for _,s in selected)})
        audits.append({'molecule_id':molecule_id,**audit})
    submission=pd.DataFrame(submissions)
    if set(submission.molecule_id)!=set(test.molecule_id) or submission.molecule_id.duplicated().any():
        raise ValueError('Submission IDs differ')
    for row in submission.itertuples():
        candidates=row.smiles.split(';')
        if not 1<=len(candidates)<=25 or len({identity(s) for s in candidates})!=len(candidates):
            raise ValueError('Invalid or official-identity duplicate candidates')
    submission.to_csv(working/'submission.csv',index=False)
    write_json(working/'submission_audit.json',audits)
    generated_summary['visible_inference']=sum(r['newgraph_count'] for r in audits)
    report={'status':'actual_conditional_fingerprint_gan_and_graph_edit_experiment',
            'generated_molecular_graphs':sum(generated_summary.values()),
            'graph_generation_by_cohort':generated_summary,
            'graph_count_unit':'unique candidate graphs per query, summed across cohorts; not globally unique molecules',
            'molecular_graph_decoder':False,
            'description':'Alternating conditional G/D fingerprint training ranks catalog structures plus a shared bounded graph-edit candidate pool.',
            'frozen_routing':{'guard':GUARD,'neural_rrf_weight':NEURAL_WEIGHT,'chemistry_weight':CHEMICAL_WEIGHT,
                              'preserve_top1_below_guard':True},
            'graph_generation_protocol':{'anchors':'up to two mass-compatible historical structures; otherwise original analog top two',
                                         'query_seed':'SHA256 of spectra/adduct/mode/precursor only, no truth identity',
                                         'shared_pool_for_supervised_and_cgan':True,
                                         'low_confidence_only':True,'max_candidates_per_query':32,
                                         'novelty_scope':'absent from this query eligible supplied pool and historical output; global PubChem novelty is not established',
                                         'method':'formula-preserving valid degree-preserving single-bond swaps'},
            'data':manifest,'training':comparison,'catalog':pool_report,
            'acceptance':acceptance_report,'known_acceptance':known_report,
            'visible_inference':{'molecules':len(submission),
              'protected':sum(r['protected'] for r in audits),'gan_changed_historical':sum(r['gan_changed_historical'] for r in audits),
              'rules_matched':sum(r['rule_matches']>0 for r in audits),
              'generated_graph_candidates':generated_summary['visible_inference']},
            'seconds':time.monotonic()-started,
            'limitations':['New experiment isolates GAN train/dev/acceptance by official identity; historical cohort overlap is not fully reconstructed.',
                           'Both arms share a wall-time ceiling, architecture, data and epoch cap; GAN adds discriminator work, so actual updates/runtime may differ. Inspect recorded histories before attributing differences solely to adversarial loss.',
                           'No learned molecular graph decoder: bounded formula-preserving graph edits provide limited new candidates; GAN generates fingerprints only.',
                           'Graph novelty is relative to each eligible supplied query pool, not verified novelty against all PubChem or the complete global chemical catalogs.',
                           'Acceptance results do not select routing or whether the explicitly requested experimental GAN is submitted.']}
    write_json(working/'gan_report.json',report)
    # Retain reviewable models, frozen protocol and measured outputs without
    # exporting full competition spectra or large reconstructable feature caches.
    write_json(working/'prepared_manifest.json',manifest)
    preprocessing_path=root/'preprocessing.json'
    if preprocessing_path.exists():
        shutil.copy2(preprocessing_path,working/'prepared_preprocessing.json')
    shutil.rmtree(root)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    return submission,report
