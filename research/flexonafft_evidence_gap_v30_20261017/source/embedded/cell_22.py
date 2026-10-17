'''--- CLAW FIX (b418/b419): stage health is now RECORDED, never silently swallowed ---'''
'''`glob('/kaggle/input/**/x')` WITHOUT recursive=True matches one directory level only, so it'''
'''misses /kaggle/input/datasets/<owner>/<slug>/. That silently disabled ICEBERG+GLACIER:'''
'''ICE FAILED -> ranker order kept: FileNotFoundError('cp313 rdkit wheel not found')  [b417 log]'''
'''rdkit-2025.3.6-cp312-...whl is not a supported wheel on this platform               [b417 log]'''
'''recursive=True finds the wheel under BOTH mount layouts (same 2025.3.6 cp313 file either way).'''
CLAW_ICE_HEALTH = 'ICE NOT RUN'
CLAW_GL_HEALTH = 'GL NOT RUN'
BASE_ICE = False         # b421: base-list ICEBERG fusion OFF (b420 showed it costs 0.009); see header
'''ICEBERG post-ranker isomer re-scoring (casmi26-iceberg). Optional: any failure keeps the ranker order.'''
import gc
ICE_SCORES, ice_stats = {}, dict(molecules=0, changed_top25=0, changed_top1=0)


'''--- CLAW FIX (b415c): make ICEBERG/GLACIER runnable on a cp313 Kaggle image ---'''
'''The ICEBERG package bundles rdkit-2025.3.6 built for **cp312**; this image is cp313,'''
'''so pip rejects that wheel and install_site() raises before anything is scored'''
'''(n_mols_scored == 0 for ICEBERG *and* GLACIER). The MANIFEST pins 2025.3.6 because'''
'''"FragmentEngine Kekulize differs in 2026.03", so swapping in the runtime's 2026.03'''
'''rdkit would change the scoring. Instead we install the SAME 2025.3.6 release built'''
'''for cp313, so the pinned version is honoured exactly and RDKIT_REQ passes natively.'''
''''''
'''install_site() is idempotent via <site>/.ice_site_ok, which it compares against the'''
'''name list of `pkg/wheels`. We pre-build the site and write that exact list, so both'''
'''runners short-circuit and never invoke the failing pip.'''
def _claw_preseed_site(pkg, site):
    import glob as _g, os as _o, shutil as _s, subprocess as _sp, sys as _sys
    srcs = sorted(_g.glob(_o.path.join(pkg, 'wheels', '*.whl.ice')) +
                  _g.glob(_o.path.join(pkg, 'wheels', '*.whl')))
    # install everything EXCEPT the cp-tag-mismatched rdkit
    keep = [w for w in srcs if not _o.path.basename(w).lower().startswith('rdkit')]
    dropped = [_o.path.basename(w) for w in srcs if w not in keep]
    new = [simulator_rdkit_wheel(pkg)]
    keep = keep + new[:1]
    _o.makedirs(site, exist_ok=True)
    wdir = _o.path.join(site, '.wheels')
    _o.makedirs(wdir, exist_ok=True)
    whl = []
    for w in keep:
        dst = _o.path.join(wdir, _o.path.basename(w)[:-4] if w.endswith('.ice') else _o.path.basename(w))
        _s.copyfile(w, dst)
        whl.append(dst)
    r = _sp.run([_sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                 '--no-compile', '--disable-pip-version-check', '--upgrade', '--target', site] + whl,
                stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
    print('[claw] site preseed rc', r.returncode, (r.stdout or '')[-400:], flush=True)
    if r.returncode != 0:
        raise RuntimeError('claw preseed pip failed: ' + (r.stdout or '')[-600:])
    _s.rmtree(wdir, ignore_errors=True)
    # the marker MUST equal exactly what install_site() derives from the PACKAGE list
    want = '\n'.join(_o.path.basename(w) for w in srcs)
    with open(_o.path.join(site, '.ice_site_ok'), 'w') as f:
        f.write(want)
    print('[claw] site ready; substituted', dropped, '->', [_o.path.basename(w) for w in new[:1]], flush=True)
    return want


try:
    ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
    sys.path.append(ICE_PKG)
    import fuse as ice_fuse
    del V, E, bank; gc.collect(); torch.cuda.empty_cache()
    # priority: molecules without a library-level match first (class 1 rarely changes), then the rest
    mids = sorted(BASE, key=lambda m: BASE[m][2])
    cands = {m: (ice_fuse.ice_candidates(BASE[m][0], BASE[m][4], TOPN) if BASE_ICE else []) for m in mids}
    if ENG:                              # ADDED (ICE_UNION): also score our engine's candidates (same-formula groups
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF   # formed over the v4g top-N + engine top-40)
        from rdkit import Chem as _Ch
        def _f2(s):
            try:
                return _CMF(_Ch.MolFromSmiles(s))
            except Exception:
                return s
        n_add = 0
        for m in mids:
            e = ENG.get(str(m))
            if not e or not e.get('smiles'):
                continue
            bs, bk, bf = list(BASE[m][0][:TOPN]), set(BASE[m][1][:TOPN]), list(BASE[m][4][:TOPN])
            es = [s for s, k in zip(e['smiles'][:40], e['keys'][:40]) if k not in bk]
            ef = [_f2(s) for s in es]
            extra = ice_fuse.ice_candidates(bs + es, bf + ef, len(bs) + len(es))
            before = len(cands.get(m, []))
            cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
            n_add += len(cands[m]) - before
        print('ICE_UNION: engine candidates added to ICE input:', n_add, flush=True)
    if ICE_PC:                          # also the PubChem-only list of molecules the v3.6 gate may use (lib_max < LIB_TAU)
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit import Chem as _Chem
        def _form(s):
            try:
                return CalcMolFormula(_Chem.MolFromSmiles(s))
            except Exception:
                return s
        for m in mids:
            p = PC.get(m) if isinstance(PC, dict) else None
            if p and p.get('pc') and BASE[m][2] < LIB_TAU:
                p['pc_form'] = [_form(x) for x in p['pc']]
                extra = ice_fuse.ice_candidates(p['pc'], p['pc_form'], 25)
                cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
    cands = {m: c for m, c in cands.items() if c}
    items = ice_fuse.build_ice_input(te, cands)
    print('ICE input molecules', len(items), 'candidates', sum(len(it['cands']) for it in items), f'{time.time()-T0:.0f}s', flush=True)
    _claw_preseed_site(ICE_PKG, '/kaggle/working/ice_site')
    ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items, workdir='/kaggle/working/ice_work', device='cuda', budget_s=ICE_BUDGET,
                                  site='/kaggle/working/ice_site', extra_args=['--any-rdkit'])
    meta_f = '/kaggle/working/ice_work/ice_out.json.meta.json'
    if os.path.exists(meta_f):
        print('ICE meta', open(meta_f).read()[:1500])
except Exception as e:
    CLAW_ICE_HEALTH = f'ICE FAILED: {e!r}'
    print('ICE FAILED -> ranker order kept:', repr(e), flush=True)
'''v4h: GLACIER third re-scoring term (casmi26-glacier) on the SAME runner input; any GLACIER failure -> v4g (ICE only)'''
GL_SCORES, gl_stats, GL_LAM, GL_BUDGET = {}, dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000
try:
    GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
    sys.path.append(GL_PKG)
    import gl_fuse
    gc.collect(); torch.cuda.empty_cache()   # ICE ran in its own (finished) process; free the notebook's cached GPU memory
    gl_items = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+']) for it in items]   # v29: GLACIER on [M+H]+ only (v27: +0.006)
    gl_items = [it for it in gl_items if it['spectra']]
    print('v29 GLACIER [M+H]+ only: molecules', len(gl_items), 'of', len(items))
    GL_SCORES = gl_fuse.run_gl(GL_PKG, gl_items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=GL_BUDGET,
                               site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'), extra_args=['--any-rdkit'])
    meta_f = '/kaggle/working/gl_work/gl_out.json.meta.json'
    print('GL meta', open(meta_f).read()[:1500] if os.path.exists(meta_f) else 'missing')
except Exception as e:
    GL_SCORES = {}
    CLAW_GL_HEALTH = f'GL FAILED: {e!r}'
    print('GL FAILED -> v4g behaviour (ICE only):', repr(e), flush=True)
def gl_of(m):
    g = GL_SCORES.get(str(m), {}) if isinstance(GL_SCORES, dict) else {}
    return g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}
def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n, tag=''):
    # z(scs) + ICE_LAM z(ice) + GL_LAM z(gl) in the same slots; no GL scores / any failure -> order0 (= v4g order)
    if not gl:
        return order0
    try:
        o = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl], [ICE_LAM, GL_LAM], top_n=top_n)
    except Exception as e:
        print('gl rerank failed', tag, repr(e)); return order0
    for k, v in (('molecules', 1), ('changed_vs_ice_top25', int(o[:25] != order0[:25])), ('changed_vs_ice_top1', int(o[:1] != order0[:1]))):
        gl_stats[tag + k] = gl_stats.get(tag + k, 0) + v
    return o
for m in list(BASE):
    smis, keys, lib_max, scs, forms = BASE[m]
    ice = ICE_SCORES.get(str(m), {}) if ICE_SCORES else {}
    gl = gl_of(m)
    if BASE_ICE and ice and any(v is not None for v in ice.values()):
        try:
            order = ice_fuse.rerank(smis, keys, scs, forms, ice, lam=ICE_LAM, top_n=TOPN)
            order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN)
            s2 = [smis[i] for i in order]; k2 = [keys[i] for i in order]
            ice_stats['molecules'] += 1
            ice_stats['changed_top25'] += int(s2[:25] != smis[:25]); ice_stats['changed_top1'] += int(bool(s2) and s2[0] != smis[0])
            smis, keys = s2, k2
        except Exception as e:
            print('rerank failed', m, repr(e))
    BASE[m] = (smis[:25], keys[:25], lib_max)
    p = PC.get(m) if (ICE_PC and isinstance(PC, dict)) else None
    if p and p.get('pc_form') and ice:
        try:                             # re-order the PubChem list by z(f.z) + lam z(ice) inside same-formula groups;
            o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
            o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')
            fz0 = p['pc_fz'][0]          # the gate keeps using the ORIGINAL top f.z (rel), so gate decisions are unchanged
            p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
            p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
            ice_stats['pc_changed'] = ice_stats.get('pc_changed', 0) + int(o[:1] != [0])
        except Exception as e:
            print('pc rerank failed', m, repr(e))
print('ICE rerank stats', ice_stats, f'{time.time()-T0:.0f}s')
print('GL rerank stats', gl_stats, f'{time.time()-T0:.0f}s')
