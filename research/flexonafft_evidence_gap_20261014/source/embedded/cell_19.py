'''all molecules -> base lists (v3.6 spectra construction and ordering)'''
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
BASE, n_err = {}, 0
POOLPOP_MU = 0.15
POOL_POP = np.load(find('pool_lsid.npy')).astype(np.float64) + np.load(find('pool_lpmid.npy')).astype(np.float64)
assert len(POOL_POP) == len(P.key), (len(POOL_POP), len(P.key))
n_pp = 0
print('v19 pool popularity re-rank, mu', POOLPOP_MU, 'pool', len(POOL_POP))
for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    smis, keys, lib_max, scs, forms = [], [], 0.0, [], []
    pids_ = []
    if len(nms):
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        try:
            C, X, F, names, info = V.run(spectra, target)
            lib_max = float(info.get('lib_max', 0.0))
            if len(C.pid):
                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                order = np.argsort(-score, kind='stable')
                seen = set()
                for i in order:
                    k = C.key[i]
                    if k in seen: continue
                    seen.add(k); smis.append(C.smiles[i]); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i]); pids_.append(int(C.pid[i]))
                    if len(smis) >= TOPN: break
        except Exception as e:
            print('ERROR', mid, repr(e)); n_err += 1
    if POOLPOP_MU > 0 and len(scs) > 1:            # v19: popularity re-rank of the top-60 base list
        try:
            s_ = np.asarray(scs, np.float64); z_ = (s_ - s_.mean()) / (s_.std() + 1e-9)
            pid_ = np.asarray(pids_); isg_ = pid_ < 0
            f_ = z_ + POOLPOP_MU * np.where(isg_, 0.0, POOL_POP[np.maximum(pid_, 0)])
            pool_i = np.where(~isg_)[0]; o_ = np.empty(len(pid_), np.int64)
            o_[isg_] = np.where(isg_)[0]; o_[~isg_] = pool_i[np.argsort(-f_[pool_i], kind='stable')]   # generated keep their slots
            smis = [smis[i] for i in o_]; keys = [keys[i] for i in o_]; scs = [float(f_[i]) for i in o_]
            forms = [forms[i] + ('|gen' if isg_[i] else '') for i in o_]      # generated form their own ICEBERG groups
            n_pp += 1
        except Exception as e:
            print('poolpop failed', mid, repr(e))
    BASE[mid] = (smis, keys, lib_max, scs, forms)
    if (gi + 1) % 25 == 0 or gi == len(mols) - 1:
        print(f'  {gi+1}/{len(mols)} molecules {time.time()-T0:.0f}s', flush=True)
print('errors', n_err, '| popularity re-ranked lists', n_pp)
'''Keep ranked overflow for append-only final coverage.'''
BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}
