ALPHA, KRR = 0.6, 3.0
CELL19_ICE = False       # b421: cell-19 ICE contribution EXPLICITLY off.
'''Reproduces BY DESIGN what b416 got by accident: when ICE errored it still wrote a'''
'''shell dict of all-None values, cell-19's `if ice:` gate passed, and ICE's column then'''
'''contributed nothing (covered count 0) -- leaving GLACIER alone. b420 (ICE real, both'''
'''applied) scored 0.402; b416 (GLACIER only) scored 0.417. This isolates that one'''
'''arithmetic difference on the healthy base.'''
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF2
from rdkit import Chem as _Ch2
def _form2(s):
    try:
        return _CMF2(_Ch2.MolFromSmiles(s))
    except Exception:
        return s
def fuse2(v_smis, e_smis, e_keys, n=40):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = chem.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]

PREFUSION_LISTS = {str(r.molecule_id): str(r.smiles).split(';') for r in pd.read_csv('submission.csv').itertuples(index=False)}
FUSION_OVERFLOW = {}
if ENG:
    v4 = pd.read_csv('submission.csv')
    changed = n_ice = 0; out = []
    for mid, s in zip(v4.molecule_id, v4.smiles):
        vs = [x for x in s.split(';') if x and x != 'CCO']
        e = ENG.get(str(mid))
        if e and e['smiles']:
            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])
            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
            if ice:
                try:
                    o = ice_fuse.rerank(fsm, fk, fsc, [_form2(x) for x in fsm], (ice if CELL19_ICE else {}), lam=ICE_LAM, top_n=len(fsm))
                    _g = gl_of(mid)
                    if _g:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [(ice if CELL19_ICE else {}), _g], [ICE_LAM, GL_LAM], top_n=len(fsm))
                    n_ice += int(o[:25] != list(range(min(25, len(o)))))
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    print('fused ICE rerank failed', mid, repr(ex))
            changed += int(bool(vs) and fsm[:1] != vs[:1])
            FUSION_OVERFLOW[str(mid)] = list(fsm[25:])
            vs = fsm[:25]
        out.append((mid, ';'.join(vs[:25]) if vs else 'CCO'))
    fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
    assert len(fs) == len(v4) and fs.molecule_id.is_unique and fs.smiles.notna().all()
    assert fs.smiles.map(lambda s: len(s.split(';'))).max() <= 25
    fs.to_csv('submission.csv', index=False)


print('V1 inference complete; applying separately audited final pass next.')
