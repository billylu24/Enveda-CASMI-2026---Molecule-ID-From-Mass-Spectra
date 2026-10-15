'''gated PubChem merge (v3.6, unchanged) -> submission.csv'''
def merge(base, base_keys, pc, pc_keys, slots, n=25):
    bk = set(base_keys)
    pcs = [s for s, k in zip(pc, pc_keys) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out
'''---- CLAW PROMOTION RULE (b417) --------------------------------------------------'''
'''The shipped gate only INTERLEAVES PubChem-only structures into fixed slots, so the'''
'''main list always keeps rank 1. When the best PubChem proposal is a strong outlier --'''
'''S = z(f.z over the pass-2 set) + 0.25*pop, carried out of the worker -- AND is well'''
'''documented (pop >= 5), let it lead; everything else keeps its usual slot.'''
'''NOTE: S is the Z-SCORED quantity (median ~6). The raw p['pc_fz'][0] (median -1437)'''
'''must never be compared to S_TAU; that would fire on ~71% of molecules.'''
'''EVIDENCE: b416 (promotion, ICEBERG broken) = 0.417 vs base415 (ICE broken, no'''
'''promotion) 0.374 and b415c (ICE working, no promotion) 0.393. The promotion overcame'''
'''losing ICE, so its marginal value is >= +0.024 (est. ~+0.037 using the notebook's own'''
'''ICE +0.013 / GL +0.006 measurements). b417 isolates it on the WORKING-ICE base.'''
S_TAU, POP_TAU = 6.0, 5.0


def promote(base, base_keys, pc, pc_keys, slots, n=25):
    """Confident PubChem proposal leads the usual list; every other entry keeps its slot."""
    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)
    if not pc:
        return usual
    top = pc[0]
    return ([top] + [x for x in usual if x != top])[:n]


'''self-test of both pure functions (this also runs in the cheap smoke pass)'''
_b, _bk = ['A', 'B', 'C'], ['ka', 'kb', 'kc']
_p, _pk = ['X', 'Y'], ['kx', 'ky']
assert merge(_b, _bk, _p, _pk, [2, 4]) == ['A', 'X', 'B', 'Y', 'C'], merge(_b, _bk, _p, _pk, [2, 4])
assert promote(_b, _bk, _p, _pk, [2, 4]) == ['X', 'A', 'B', 'Y', 'C'], promote(_b, _bk, _p, _pk, [2, 4])
assert merge(_b, _bk, _p, _pk, [2, 4])[0] == 'A', 'shipped gate must keep rank 1'
assert merge(_b, _bk, [], [], [2, 4]) == ['A', 'B', 'C']
assert promote(_b, _bk, [], [], [2, 4]) == ['A', 'B', 'C']
assert promote(_b, _bk, _p, _pk, [2, 4])[:1] == ['X'] and len(promote(_b, _bk, _p, _pk, [2, 4])) == 5
print('merge/promote self-test OK')

rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0, promoted=0)
for mid in te.molecule_id.unique():
    smis, keys, lib_max = BASE.get(mid, ([], [], 0.0))
    p = PC.get(mid)
    if p is None or not p['pc']:
        stats['no_pc'] += 1; final = smis
    elif lib_max >= LIB_TAU:
        stats['untouched'] += 1; final = smis
    else:
        rel = (p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
        slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
        stats['aggressive' if rel > REL_TH else 'gentle'] += 1
        S = p.get('S'); pop_top = float(p.get('top_pop') or 0.0)
        # Skomuro LB 0.403 Rule: Promote on Popularity OR Structure-Evidence Margin!
        # rel > 250 captures confident PubChem outliers that popularity alone misses
        if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > 250.0):
            stats['promoted'] += 1
            final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
        else:
            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)
    if not final: final = ['CCO']
    rows.append((mid, ';'.join(final[:25])))
print('merge stats', stats)
sub = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
sub = samp[['molecule_id']].merge(sub, on='molecule_id', how='left')
sub['smiles'] = sub['smiles'].fillna('CCO')
assert len(sub) == len(samp) and sub.molecule_id.is_unique and sub.smiles.notna().all()
assert sub.smiles.map(lambda s: len(s.split(';'))).max() <= 25
sub.to_csv('submission.csv', index=False)
print('CLAW STAGE HEALTH |', CLAW_ICE_HEALTH, '|', CLAW_GL_HEALTH, flush=True)
try:
    _ice_n = json.load(open('/kaggle/working/ice_work/ice_out.json.meta.json')).get('n_mols_scored', 0)
    _gl_n = json.load(open('/kaggle/working/gl_work/gl_out.json.meta.json')).get('n_mols_scored', 0)
except Exception:
    _ice_n = _gl_n = -1
print(f'CLAW STAGE SCORED | ice={_ice_n} gl={_gl_n}', flush=True)
json.dump(dict(secs=round(time.time() - T0), merge=stats, n_errors=n_err,
               ice_health=CLAW_ICE_HEALTH, gl_health=CLAW_GL_HEALTH,
               ice_n_scored=_ice_n, gl_n_scored=_gl_n,
               families=MAN['families'],
               gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else None), open('run_manifest.json', 'w'), indent=1)
print('done', sub.shape, f'{time.time()-T0:.0f}s')
sub.head()
