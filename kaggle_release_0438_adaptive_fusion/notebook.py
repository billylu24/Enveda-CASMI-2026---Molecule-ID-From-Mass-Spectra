# %% [markdown]
# CASMI26 Adaptive Simulator Fusion (0.438 -> 0.44x Upgrade)
# 
# Source: samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438, public V1, retrieved October 10, 2026. All source credits retained. The title's 0.438 is an upstream claim, not our measured score. Original model, weights, thresholds, 80-candidate fusion window, metric-safe final tail, and original hidden-test inference are retained. Original 12-molecule commit smoke and full hidden rerun are preserved. Added assertions verify all active molecules and stage completion; hidden rerun rejects unprocessed rows. Public data overlaps training; no generalization score is claimed from visible data.


# %% [markdown]
# casmi-v14 = v13 + ICEBERG rerank for molecules without GLACIER scores (no [M+H]+ spectra)
# casmi-v13 = casmi-v11 (LB 0.433) with the fused list reranked by GLACIER widened 40 -> 80
# 
# CASMI26 V1.1 — metric audit and safe tail
# 
# Starting point: this account's V1 submission, public MRR **0.425** on 2026-10-08.
# This is a conservative candidate-coverage experiment, not a claimed leaderboard improvement.
# 
# Changes: pin official RDKit 2026.03.3; preserve the V1 inference/ranking pipeline;
# deduplicate only after ranking using the official metric identity; append ranked overflow
# and mass-compatible DreaMS candidates into vacant slots; assert every original matchable
# identity has equal or better rank. The old DreaMS branch was bypassed by `NameError: Path`.
# 
# No hidden-test labels, manual labels, or network access are used. The visible test overlaps
# training data, so its score is an operational check and is **not** a validation estimate.
# 
# Source attribution: copied V1 ensemble from BobtheBot369 / Ahmed Berat Özer / prvsiyan
# and the other authors credited in its code. Scoring reference: Kaggle Competition Metrics,
# `https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank`, version 14, Apache-2.0.


# %% [markdown]
## 1. Setup
# 
# Locates every dataset, installs the competition's RDKit wheel, sets the popularity-prior parameters for the PubChem channel (`POP_LAM = 0.25`, `POP_UNION = 200`) and checks the v4b model files against their manifest.


# %% [code]
# CASMI26 v4n = v4i + ICE_LAM = GL_LAM = 1.0 (v4l, LB 0.380) + engine FP bank fpnet_full1 (v4m, LB 0.378)
# CASMI26 v4i = v4h + PubChem-only channel pass-1 depth N1 1000 -> 5000 (recall; gate/slots unchanged)
# CASMI26 v4h = v4g + GLACIER third re-scoring term: z(ranker) + 0.5 z(ICEBERG) + 0.5 z(GLACIER) (casmi26-glacier)
# casmi26 v28 = v17 + pool popularity re-rank of the base list (mu 0.15, mode A); v17 = Ahmed v4g + popularity prior in the PubChem list (see make_v17.py)
# CASMI26 v4f = v4b (FPNet A engine bank, LB 0.354) + ICEBERG post-ranker isomer re-scoring = the v3.6 model (v1 engine + FPNet A + moderate generation + gated PubChem channel, LB 0.358)
# + fe_v4 feature families computed on the SAME v1 engine run (derivation evidence / class-3 priors, fragmentation 2.0,
# analog-structure relations, DreamsFP views) + a ranker retrained on the v1-engine simulation sim1x with them.
# Built by work/analysis/fe_v4/build_v4v1.py -- edit there.
import os, sys, glob, re, subprocess, time, json, hashlib, pickle
from pathlib import Path
T0 = time.time()
LIB_TAU, REL_TH = 0.9, 600.0
TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True
SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]
os.makedirs('/kaggle/working/numba_cache', exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = '/kaggle/working/numba_cache'
def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]
def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(n), b''):
            h.update(b)
    return h.hexdigest()
POP_LAM, POP_UNION = 0.25, 200            # v17: popularity prior for the PubChem-only list
os.environ['CASMI_POP_DIR'] = os.path.dirname(find('pc_lsid.npy')); os.environ['CASMI_POP_LAM'] = str(POP_LAM); os.environ['CASMI_POP_UNION'] = str(POP_UNION)
print('v17 popularity prior', POP_LAM, POP_UNION, os.environ['CASMI_POP_DIR'])
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w and Path(w).name.startswith('rdkit-2026.3.3-')]
assert whl, 'Official RDKit 2026.3.3 wheel is missing'
print('rdkit wheel:', whl)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
r.check_returncode()
import rdkit; print('rdkit', rdkit.__version__)
assert rdkit.__version__ == '2026.03.3'
V4CODE = os.path.dirname(os.path.dirname(find('casmi26-v4b-models/**/casmi/engine.py')))
V4MOD = os.path.dirname(find('casmi26-v4b-models/**/MANIFEST.json'))
V3CODE = os.path.dirname(os.path.dirname(find('casmi26-v3-models/**/casmi/engine.py')))
V3MOD = os.path.dirname(find('casmi26-v3-models/**/ranker_0.pkl'))
POOL_DIR = os.path.dirname(find('pool_meta.parquet'))
PC_DIR = os.path.dirname(find('pc_smiles.npy'))
COMP = os.path.dirname(find('test.parquet'))
MAN = json.load(open(os.path.join(V4MOD, 'MANIFEST.json')))
bad = []
for rel, h in sorted(MAN['files'].items()):
    p = os.path.join(V4CODE, rel[len('code/'):]) if rel.startswith('code/') else os.path.join(V4MOD, rel)
    if not os.path.exists(p) or sha256(p) != h:
        bad.append(rel)
assert not bad, f'dataset files differ from MANIFEST.json: {bad}'
print('V4CODE', V4CODE, '\nV4MOD', V4MOD, '\nV3CODE', V3CODE, '\nPOOL', POOL_DIR, '\nPC', PC_DIR, '\nCOMP', COMP)
print(f"{len(MAN['files'])} files match MANIFEST.json; families {MAN['families']}", f'{time.time()-T0:.0f}s')

# %% [markdown]
## 2. Commit switch
# 
# When the visible placeholder test is detected, the notebook runs only 12 molecules with a short ICEBERG budget, so saving a version is quick. The scoring rerun on the hidden test runs everything. Nothing else depends on this switch.


# %% [code]
# SMOKE switch: commits (dummy test) run 12 molecules; the scoring rerun (hidden test) runs everything.
import pandas as pd, hashlib
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_sig = hashlib.md5(','.join(sorted(_te.spectrum_id.astype(str))).encode()).hexdigest()
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'
SMOKE_N = 12
if not IS_RERUN:
    SM = '/kaggle/working/smoke_comp'; os.makedirs(SM, exist_ok=True)
    _full = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _keep = sorted(_full.molecule_id.unique())[:SMOKE_N]
    _full[_full.molecule_id.isin(_keep)].to_parquet(os.path.join(SM, 'test.parquet'))
    for _f in ['train.parquet', 'sample_submission.csv']:
        if not os.path.exists(os.path.join(SM, _f)):
            os.symlink(os.path.join(COMP, _f), os.path.join(SM, _f))
    COMP = SM
    ICE_BUDGET = 300
print('RERUN', IS_RERUN, '| sig', _sig, '| COMP', COMP, '| ICE_BUDGET', ICE_BUDGET)


# %% [markdown]
## 3. Engine 2: two-ranker engine (BIO + AFIX)
# 
# This engine runs first in its own process, while GPU memory is still empty. It writes a top-40 list per molecule with tautomer-canonical InChIKey-14 keys.
# 
# - **Candidates** come from COCONUT, ChEBI/LIPID MAPS (BIO) and the training structures.
# - **Evidence** comes from library matching (including adduct-shifted matches), analog propagation, prvsiyan's fingerprint models and MetFrag-lite fragment matching.
# - **Ranking** blends two gradient-boosted rankers.


# %% [code]
# Our engine (two-ranker + BIO + AFIX) in a subprocess, then weighted reciprocal-rank fusion with the v4g lists.
ENG_DIR = '/kaggle/working/eng'; os.makedirs(ENG_DIR, exist_ok=True)
ENG_FILES = {'pv.py': '"""prvsiyan analog-propagation pipeline components (public notebook, version of 2026-09-16), copied\nverbatim where possible so local simulations and the Kaggle notebook share one source of truth.\nAdditions are marked `# ADDED`.\n"""\nimport math\nimport numpy as np\nfrom numba import njit, prange\n\n\nclass CFG:\n    PPM_WIN = 10.0\n    PPM_FALLBACK = 30.0\n    INT_FLOOR = 0.002\n    MAX_PEAKS = 256\n    MZ_TOL = 0.01\n    INT_POWER = 1.0\n    ENT_WEIGHT = True\n    ANALOG_WIN = 200.0\n    N_ANALOG = 80\n    SIM_POWER = 3.0\n    W1_PRIORS = (0.30, 0.60)\n    SEEDS = (0, 1, 2, 3)\n    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)\n    TOPN = 25\n\n\n# ----------------------------------------------------------------------------------------------\n# similarity kernels\n# ----------------------------------------------------------------------------------------------\n@njit(cache=True, fastmath=True)\ndef _clean(mz, it, floor, topk, power, ent_weight):\n    n = len(mz)\n    if n == 0: return np.empty(0, np.float32), np.empty(0, np.float32)\n    mx = 0.0\n    for i in range(n):\n        if it[i] > mx: mx = it[i]\n    if mx <= 0: return np.empty(0, np.float32), np.empty(0, np.float32)\n    thr = floor * mx; c = 0\n    for i in range(n):\n        if it[i] >= thr: c += 1\n    idx = np.empty(c, np.int64); j = 0\n    for i in range(n):\n        if it[i] >= thr: idx[j] = i; j += 1\n    if c > topk:\n        v = np.empty(c, np.float32)\n        for i in range(c): v[i] = it[idx[i]]\n        o = np.argsort(v)[c - topk:]\n        k2 = np.empty(topk, np.int64)\n        for i in range(topk): k2[i] = idx[o[i]]\n        k2.sort(); idx = k2; c = topk\n    om = np.empty(c, np.float32); oi = np.empty(c, np.float32)\n    s = 0.0\n    for i in range(c):\n        om[i] = mz[idx[i]]; v = it[idx[i]] ** power; oi[i] = v; s += v\n    if s > 0:\n        for i in range(c): oi[i] /= s\n    if ent_weight:\n        S = 0.0\n        for i in range(c):\n            if oi[i] > 0: S -= oi[i] * np.log(oi[i])\n        if S < 3.0:\n            w = 0.25 + 0.25 * S; s2 = 0.0\n            for i in range(c): oi[i] = oi[i] ** w; s2 += oi[i]\n            if s2 > 0:\n                for i in range(c): oi[i] /= s2\n    return om, oi\n\n\n@njit(cache=True, fastmath=True)\ndef entropy_sim(qmz, qp, cmz, cp, tol):\n    i = 0; j = 0; n = len(qmz); m = len(cmz)\n    SA = 0.0\n    for x in range(n):\n        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])\n    SB = 0.0\n    for x in range(m):\n        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])\n    SAB = 0.0; tot = 0.0\n    buf = np.empty(n + m, np.float64); b = 0\n    while i < n and j < m:\n        d = qmz[i] - cmz[j]\n        if d < -tol: buf[b] = qp[i]; i += 1; b += 1\n        elif d > tol: buf[b] = cp[j]; j += 1; b += 1\n        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1\n    while i < n: buf[b] = qp[i]; i += 1; b += 1\n    while j < m: buf[b] = cp[j]; j += 1; b += 1\n    for x in range(b): tot += buf[x]\n    if tot <= 0: return 0.0\n    for x in range(b):\n        v = buf[x] / tot\n        if v > 0: SAB -= v * np.log(v)\n    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)\n\n\n@njit(cache=True, fastmath=True, parallel=True)\ndef search(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):\n    out = np.zeros(len(cand), np.float32)\n    for k in prange(len(cand)):\n        c = cand[k]; a = off[c]; b = off[c + 1]\n        if b <= a: continue\n        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)\n        if len(cm) == 0: continue\n        out[k] = entropy_sim(qmz, qp, cm, cp, tol)\n    return out\n\n\n@njit(cache=True, fastmath=True)\ndef entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):\n    a = entropy_sim(qmz, qp, cmz, cp, tol)\n    if shift > -0.001 and shift < 0.001: return a\n    sm = np.empty(len(cmz), np.float32)\n    for i in range(len(cmz)): sm[i] = cmz[i] + shift\n    b = entropy_sim(qmz, qp, sm, cp, tol)\n    return a if a > b else b\n\n\n# ADDED: search over a pre-cleaned store (reps cleaned once instead of on every call)\n@njit(cache=True, fastmath=True, parallel=True)\ndef search_shift_pre(qmz, qp, lo, hi, roff, rmz, rit, tol, shift):\n    out = np.zeros(hi - lo, np.float32)\n    for k in prange(hi - lo):\n        a = roff[lo + k]; b = roff[lo + k + 1]\n        if b <= a: continue\n        out[k] = entropy_sim_shift(qmz, qp, rmz[a:b], rit[a:b], tol, shift[k])\n    return out\n\n\n# ADDED: library search where each reference is shifted by (query precursor - reference precursor):\n# the same molecule measured as a different adduct keeps its neutral losses (modified-cosine idea)\n@njit(cache=True, fastmath=True, parallel=True)\ndef search_shift_rows(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shift):\n    out = np.zeros(len(cand), np.float32)\n    for k in prange(len(cand)):\n        c = cand[k]; a = off[c]; b = off[c + 1]\n        if b <= a: continue\n        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)\n        if len(cm) == 0: continue\n        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shift[k])\n    return out\n\n\n# ADDED: clean a list of spectra into a flat store\n@njit(cache=True)\ndef clean_store(rows, off, allmz, allin, floor, topk, power, ent_weight):\n    n = len(rows)\n    lens = np.zeros(n + 1, np.int64)\n    tmp_m = []\n    tmp_i = []\n    for k in range(n):\n        r = rows[k]\n        cm, cp = _clean(allmz[off[r]:off[r + 1]], allin[off[r]:off[r + 1]], floor, topk, power, ent_weight)\n        tmp_m.append(cm); tmp_i.append(cp)\n        lens[k + 1] = lens[k] + len(cm)\n    om = np.empty(lens[n], np.float32); oi = np.empty(lens[n], np.float32)\n    for k in range(n):\n        om[lens[k]:lens[k + 1]] = tmp_m[k]; oi[lens[k]:lens[k + 1]] = tmp_i[k]\n    return lens, om, oi\n\n\n# ----------------------------------------------------------------------------------------------\n# adducts\n# ----------------------------------------------------------------------------------------------\nMASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,\n            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,\n            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)\nE = 0.00054857990; PROTON = MASS[\'H\'] - E; H2O = 2 * MASS[\'H\'] + MASS[\'O\']\nNH4 = MASS[\'N\'] + 4 * MASS[\'H\']; FORMATE = MASS[\'C\'] + 2 * MASS[\'H\'] + 2 * MASS[\'O\']\nACETATE = 2 * MASS[\'C\'] + 4 * MASS[\'H\'] + 2 * MASS[\'O\']\nADDUCTS = {\n    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS[\'Na\'] - E),\n    "[M+K]+": (1, 1, MASS[\'K\'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),\n    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),\n    "[M+CH3OH+H]+": (1, 1, PROTON + MASS[\'C\'] + 4 * MASS[\'H\'] + MASS[\'O\']),\n    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS[\'C\'] + 3 * MASS[\'H\'] + MASS[\'N\']),\n    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),\n    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS[\'Cl\'] + E), "[M]-": (1, 1, E),\n    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS[\'Na\'] - 2 * PROTON),\n    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS[\'Na\'] - E), "[2M+NH4]+": (2, 1, NH4 - E),\n    "[2M+K]+": (2, 1, MASS[\'K\'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),\n    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS[\'Na\'] - 2 * PROTON),\n    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),\n}\n\n\ndef neutral_mass(mz, adduct):\n    out = np.full(len(mz), np.nan); ad = np.asarray(adduct, dtype=object)\n    for a, (n, z, d) in ADDUCTS.items():\n        m = (ad == a)\n        if m.any(): out[m] = (mz[m] * z - d) / n\n    return out\n\n\n# ----------------------------------------------------------------------------------------------\n# ranking features (31)\n# ----------------------------------------------------------------------------------------------\nN_ANALOG = 80\nP_SIM = 3.0\nN_FEAT = 31\n\n\ndef _rank_norm(x):\n    o = np.argsort(-x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)\n\n\ndef _z(x):\n    s = x.std()\n    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)\n\n\ndef rank_features(cand_fp, cand_lib, analog_fp, analog_sim, model_logits=None, frag=None):\n    nc = cand_fp.shape[0]\n    cf = cand_fp.astype(np.float32); cs = cf.sum(1)\n    lv = np.asarray(cand_lib, np.float32)\n    lvmax = float(lv.max()) if nc else 0.0\n    if analog_fp is not None and len(analog_sim):\n        af = analog_fp.astype(np.float32); asum = af.sum(1)\n        inter = cf @ af.T\n        tan = inter / (cs[:, None] + asum[None, :] - inter + 1e-9)\n        w = np.clip(np.asarray(analog_sim, np.float32), 0, None)\n        ap = (tan * (w ** P_SIM)[None, :]).max(1)\n        a1 = (tan * w[None, :]).max(1)\n        best_tan = tan.max(1); top_tan = tan[:, 0]; top_sim = float(w[0])\n        mean_tan = (tan * (w ** P_SIM)[None, :]).sum(1) / ((w ** P_SIM).sum() + 1e-9)\n    else:\n        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32); top_sim = 0.0\n    apmax = float(ap.max()) if nc else 0.0\n    if model_logits is not None:\n        raw = cf @ np.asarray(model_logits, np.float32)\n        nrm = raw / np.sqrt(np.maximum(cs, 1.0))\n        mfeat = [_z(raw), _rank_norm(raw), raw - raw.max(), _z(nrm), _rank_norm(nrm),\n                 (raw == raw.max()).astype(np.float32)]\n    else:\n        mfeat = [np.zeros(nc, np.float32)] * 6\n    if model_logits is not None and nc:\n        mr = _rank_norm(cf @ np.asarray(model_logits, np.float32))\n        lbest = int(np.argmax(lv)) if lvmax > 0 else -1\n        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0\n        abest = int(np.argmax(ap)) if apmax > 0 else -1\n        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0\n        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),\n                 np.full(nc, agree * lvmax), np.full(nc, float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0)]\n    else:\n        xfeat = [np.zeros(nc, np.float32)] * 6\n    if frag is not None:\n        fr = np.asarray(frag, np.float32)\n        ffeat = [fr, _rank_norm(fr), fr - fr.max() if nc else fr, _z(fr)]\n    else:\n        ffeat = [np.zeros(nc, np.float32)] * 4\n    return np.column_stack([\n        lv, _rank_norm(lv), np.full(nc, lvmax), lv - lvmax, (lv > 0).astype(float),\n        ap, _rank_norm(ap), np.full(nc, apmax), ap - apmax,\n        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim),\n        np.full(nc, np.log(max(nc, 1))),\n        *mfeat, *ffeat, *xfeat,\n    ]).astype(np.float32)\n\n\n# ----------------------------------------------------------------------------------------------\n# MetFrag-lite\n# ----------------------------------------------------------------------------------------------\nAMU = {\'C\': 12.0, \'H\': 1.00782503207, \'N\': 14.0030740048, \'O\': 15.9949146196, \'P\': 30.97376163,\n       \'S\': 31.97207100, \'F\': 18.99840322, \'Cl\': 34.96885268, \'Br\': 78.9183371, \'I\': 126.904473,\n       \'Na\': 22.9897692809, \'K\': 38.96370668, \'Si\': 27.9769265325, \'B\': 11.0093054, \'Se\': 79.9165213}\nH = AMU[\'H\']\n\n\ndef mol_graph(smi):\n    from rdkit import Chem, RDLogger\n    RDLogger.DisableLog(\'rdApp.*\')\n    m = Chem.MolFromSmiles(smi)\n    if m is None: return None\n    n = m.GetNumAtoms()\n    w = np.zeros(n)\n    for a in m.GetAtoms():\n        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H\n    if (w == 0).any(): return None\n    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]\n    return w, bonds, n\n\n\ndef _components(n, bonds, drop):\n    adj = [[] for _ in range(n)]\n    for i, (a, b) in enumerate(bonds):\n        if i in drop: continue\n        adj[a].append(b); adj[b].append(a)\n    seen = np.zeros(n, bool); comps = []\n    for s in range(n):\n        if seen[s]: continue\n        stack = [s]; seen[s] = True; cur = [s]\n        while stack:\n            u = stack.pop()\n            for v in adj[u]:\n                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)\n        comps.append(cur)\n    return comps\n\n\ndef fragment_masses(smi, max_breaks=2, max_bonds=34):\n    g = mol_graph(smi)\n    if g is None: return np.zeros(0)\n    w, bonds, n = g\n    nb = len(bonds)\n    if nb == 0 or nb > max_bonds: return np.array([w.sum()])\n    out = {w.sum()}\n    for i in range(nb):\n        for c in _components(n, bonds, {i}):\n            out.add(float(w[c].sum()))\n    if max_breaks >= 2:\n        for i in range(nb):\n            for j in range(i + 1, nb):\n                for c in _components(n, bonds, {i, j}):\n                    out.add(float(w[c].sum()))\n    return np.array(sorted(out))\n\n\ndef frag_masses_safe(smi):\n    try: return fragment_masses(smi)\n    except Exception: return np.zeros(0)\n\n\n# ADDED: charge carriers a fragment may keep, per precursor adduct (prvsiyan only used +-proton)\nCARRIERS = {\n    "[M+H]+": (PROTON,), "[M-H2O+H]+": (PROTON,), "[M-2H2O+H]+": (PROTON,),\n    "[M+NH4]+": (PROTON, NH4 - E), "[M+Na]+": (PROTON, MASS[\'Na\'] - E), "[M+K]+": (PROTON, MASS[\'K\'] - E),\n    "[M-H]-": (-PROTON,), "[M-H2O-H]-": (-PROTON,), "[M+CH2O2-H]-": (-PROTON, FORMATE - PROTON),\n    "[M+Cl]-": (-PROTON, MASS[\'Cl\'] + E),\n}\n\n\ndef explain_score_adduct(frag_mass, peak_mz, peak_int, adduct, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):\n    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0\n    car = CARRIERS.get(adduct, (PROTON,) if "+" in adduct[-2:] else (-PROTON,))\n    ion = np.sort(np.concatenate([frag_mass + dh * H + c for dh in h_shifts for c in car]))\n    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()\n    if tot <= 0: return 0.0\n    idx = np.searchsorted(ion, peak_mz)\n    ok = np.zeros(len(peak_mz), bool)\n    for off in (-1, 0):\n        k = np.clip(idx + off, 0, len(ion) - 1)\n        ok |= np.abs(ion[k] - peak_mz) <= tol\n    return float(w[ok].sum() / tot)\n\n\ndef explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):\n    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0\n    ion = []\n    for dh in h_shifts:\n        ion.append(frag_mass + dh * H + (PROTON if mode > 0 else -PROTON))\n    ion = np.sort(np.concatenate(ion))\n    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()\n    if tot <= 0: return 0.0\n    idx = np.searchsorted(ion, peak_mz)\n    ok = np.zeros(len(peak_mz), bool)\n    for off in (-1, 0):\n        k = np.clip(idx + off, 0, len(ion) - 1)\n        ok |= np.abs(ion[k] - peak_mz) <= tol\n    return float(w[ok].sum() / tot)', 'pv_fp.py': '"""Spectrum -> fingerprint model (prvsiyan FPNet), kept apart so CPU worker processes never import torch."""\nimport math\nimport numpy as np\n# ----------------------------------------------------------------------------------------------\n# spectrum -> fingerprint model\n# ----------------------------------------------------------------------------------------------\nimport torch, torch.nn as nn, torch.nn.functional as F\n\nFP_MAX_PEAKS = 128\nADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",\n               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",\n               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",\n               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]\nADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}\nINSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]\n\n\ndef instr_family(s):\n    if s is None: return 4\n    t = str(s).lower()\n    if \'timstof\' in t: return 0\n    if \'orbitrap\' in t or \'qft\' in t or \'ftms\' in t or \'hybrid ft\' in t or \'itft\' in t or \'exactive\' in t: return 1\n    if \'tof\' in t: return 2\n    if \'trap\' in t or \'qq\' in t: return 3\n    return 4\n\n\ndef prep_peaks(mz, inten, prec_mz, max_peaks=FP_MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):\n    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)\n    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    keep = (mz <= prec_mz + 1.5)\n    mz, it = mz[keep], it[keep]\n    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    mx = it.max()\n    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)\n    keep = it >= floor * mx\n    mz, it = mz[keep], it[keep]\n    if len(mz) > max_peaks:\n        order = np.argsort(-it)\n        bucket = (mz // win).astype(np.int64)\n        cnt = {}; sel = []\n        for i in order:\n            b = bucket[i]; c = cnt.get(b, 0)\n            if c < per_win: cnt[b] = c + 1; sel.append(i)\n        sel = np.array(sel)\n        if len(sel) > max_peaks:\n            sel = sel[np.argsort(-it[sel])[:max_peaks]]\n        elif len(sel) < max_peaks:\n            rest = np.array([i for i in order if i not in set(sel.tolist())])\n            need = max_peaks - len(sel)\n            if len(rest): sel = np.concatenate([sel, rest[:need]])\n        mz, it = mz[sel], it[sel]\n    o = np.argsort(mz)\n    mz, it = mz[o], it[o]\n    v = np.sqrt(it / it.max())\n    return mz.astype(np.float32), v.astype(np.float32)\n\n\nclass SinEmb(nn.Module):\n    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):\n        super().__init__()\n        n = dim // 2\n        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)\n        self.register_buffer(\'inv\', (2 * math.pi) / wav)\n\n    def forward(self, x):\n        a = x.unsqueeze(-1) * self.inv\n        return torch.cat([torch.sin(a), torch.cos(a)], -1)\n\n\nclass Block(nn.Module):\n    def __init__(self, d, h, drop):\n        super().__init__(); self.h = h\n        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)\n        self.n2 = nn.LayerNorm(d)\n        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))\n        self.drop = nn.Dropout(drop)\n\n    def forward(self, x, pad):\n        B, N, D = x.shape; y = self.n1(x)\n        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)\n        m = (~pad)[:, None, None, :]\n        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)\n        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))\n        return x + self.drop(self.ff(self.n2(x)))\n\n\nclass FPNet(nn.Module):\n    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):\n        super().__init__()\n        self.d = d\n        self.mz_emb = SinEmb(d)\n        self.nl_emb = SinEmb(d)\n        self.pk = nn.Linear(2 * d + 1, d)\n        self.prec_emb = SinEmb(d)\n        self.ad = nn.Embedding(len(ADDUCT_LIST), d)\n        self.ins = nn.Embedding(len(INSTR_LIST), d)\n        self.gl = nn.Linear(d + 3, d)\n        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])\n        self.norm = nn.LayerNorm(d)\n        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))\n\n    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):\n        B, N = mz.shape\n        nl = (prec[:, None] - mz).clamp(min=0)\n        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))\n        g = self.gl(torch.cat([self.prec_emb(prec),\n                               (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),\n                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)\n        x = torch.cat([g.unsqueeze(1), p], 1)\n        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)\n        for b in self.blocks: x = b(x, pad)\n        x = self.norm(x)\n        cls = x[:, 0]\n        msk = (~pad[:, 1:]).float().unsqueeze(-1)\n        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)\n        return self.head(torch.cat([cls, mean], -1))\n\n\ndef load_fp_models(paths, dev):\n    single, merged = [], []\n    nbits = None\n    for pth in paths:\n        ck = torch.load(pth, map_location=\'cpu\', weights_only=False)\n        net = FPNet(ck[\'nbits\'], d=ck[\'d\'], layers=ck[\'layers\']).to(dev).eval()\n        net.load_state_dict(ck[\'model\'])\n        (merged if \'merged\' in str(pth).replace(\'\\\\\', \'/\').split(\'/\')[-1] else single).append(net)\n        nbits = ck[\'nbits\']\n    return single, merged, nbits\n\n\ndef merge_peaks(specs):\n    """specs: list of (mz, it). All peaks collapsed into one pseudo-spectrum."""\n    specs = [(a, b) for a, b in specs if len(a)]\n    if not specs: return np.zeros(0), np.zeros(0)\n    mz = np.concatenate([np.asarray(a, float) for a, _ in specs])\n    it = np.concatenate([np.asarray(b, float) / max(float(np.asarray(b, float).max()), 1e-9) for _, b in specs])\n    o = np.argsort(mz); mz, it = mz[o], it[o]\n    keep = np.ones(len(mz), bool)\n    for j in range(1, len(mz)):\n        if mz[j] - mz[j - 1] < 0.005:\n            if it[j] >= it[j - 1]: keep[j - 1] = False\n            else: keep[j] = False\n    return mz[keep], it[keep]\n\n\n@torch.no_grad()\ndef logits_batch(P, nets, dev, prec, adduct, instr, ce, mode):\n    """P: list of (mz, it) already prepped; per-row arrays prec/adduct(str)/instr(str)/ce/mode."""\n    keep = [i for i, (a, _) in enumerate(P) if len(a)]\n    if not keep: return None\n    P = [P[i] for i in keep]\n    B = len(P); N = max(len(a) for a, _ in P)\n    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)\n    for i, (a, b) in enumerate(P):\n        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False\n    T = lambda x: torch.as_tensor(x, device=dev)\n    args = (T(mz), T(it), T(pad),\n            T(np.asarray(prec, np.float32)[keep]),\n            T(np.array([ADDUCT_IX.get(a, ADDUCT_IX[\'<unk>\']) for a in np.asarray(adduct, object)[keep]])),\n            T(np.array([instr_family(s) for s in np.asarray(instr, object)[keep]])),\n            T(np.asarray(ce, np.float32)[keep]),\n            T(np.asarray(mode, np.float32)[keep]))\n    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)\n\n\ndef molecule_logits(models, specs, prec, adduct, instr, ce, mode):\n    """prvsiyan model_logits(): mean of (single-model per-spectrum mean) and (merged-model on merged list).\n    specs: list of (mz, it); prec/adduct/instr/ce/mode: per-spectrum."""\n    single, merged, dev = models\n    out = []\n    if single:\n        P = [prep_peaks(a, b, p) for (a, b), p in zip(specs, prec)]\n        za = logits_batch(P, single, dev, prec, adduct, instr, ce, mode)\n        if za is not None: out.append(za)\n    if merged:\n        mz, it = merge_peaks(specs)\n        pm = float(np.median(prec))\n        P = [prep_peaks(mz, it, pm)]\n        zb = logits_batch(P, merged, dev, [pm], [adduct[0]], [instr[0]], [25.0], [float(np.mean(mode))])\n        if zb is not None: out.append(zb)\n    return np.mean(out, axis=0) if out else None', 'casmi_engine.py': '"""End-to-end inference (Kaggle notebook body). Mirrors e02_channels.py + e05_rank.py feature blocks exactly,\nbut reads train.parquet / test.parquet directly. Runs locally too (set CASMI_LOCAL=1).\n\nStages: library -> candidate pool -> index -> per-molecule channels -> MetFrag-lite -> features -> ranker -> submission\n"""\nimport os, sys, glob, time, pickle, math\nos.environ.setdefault("OPENBLAS_NUM_THREADS", "1")\nimport numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa\nimport pv\nfrom pv import CFG\nimport re as _re\n_MONO = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100, P=30.97376163,\n             Cl=34.96885268, Br=78.9183371, F=18.99840322, I=126.904473, Si=27.9769265325, Se=79.9165213,\n             Na=22.9897692809, K=38.96370668, B=11.0093054, As=74.9215965, Fe=55.9349375, D=2.0141017778)\n_TOK = _re.compile(r"([A-Z][a-z]?)(\\d*)")\n\n\ndef formula_mass(f):\n    """ADDED: neutral monoisotopic mass by arithmetic on molecular_formula (prvsiyan fix; immune to\n    riken\'s 0.005 Da precursor precision and gnps adduct mislabels). NaN if an element is untabulated."""\n    if not isinstance(f, str) or not f: return np.nan\n    m = 0.0\n    for el, n in _TOK.findall(f):\n        if el not in _MONO: return np.nan\n        m += _MONO[el] * (int(n) if n else 1)\n    return m\n\nPOOL = None\n\n\nclass RANK:\n    W_A = (0.35, 0.55)      # weight on class-1 simulation rows; two priors averaged\n    SEEDS = (0, 1)\n    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)\n\nT0 = time.time()\nLOCAL = os.environ.get("CASMI_LOCAL") == "1"\nROOTS = [r"C:\\Users\\HW-LEE\\Desktop\\CASMI"] if LOCAL else ["/kaggle/input"]\n\n\ndef find(name):\n    for root in ROOTS:\n        hits = glob.glob(os.path.join(root, "**", name), recursive=True)\n        if hits: return sorted(hits, key=len)[0]\n    raise FileNotFoundError(name)\n\n\ndef log(msg):\n    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)\n\n\n# ---------------------------------------------------------------------------------------------\n# library\n# ---------------------------------------------------------------------------------------------\ndef load_library(path):\n    """Row group by row group, peaks straight to float32: peak RAM stays near the final ~3.5 GB."""\n    f = pq.ParquetFile(path)\n    n_rows = f.metadata.num_rows\n    npk = 0\n    for i in range(f.num_row_groups):\n        c = f.read_row_group(i, columns=["ms2_mzs"]).column(0).combine_chunks()\n        npk += len(c.values)\n    off = np.zeros(n_rows + 1, np.int64)\n    allmz = np.empty(npk, np.float32); allin = np.empty(npk, np.float32)\n    prec, add, pol, ik, smi = [], [], [], [], []\n    fo, tims = [], []\n    r0 = p0 = 0\n    for i in range(f.num_row_groups):\n        t = f.read_row_group(i, columns=["inchikey14", "normalized_smiles", "adduct", "precursor_mz", "ionization_mode",\n                                         "ms2_mzs", "ms2_normalized_intensities", "molecular_formula", "instrument_type"])\n        mzc = t.column("ms2_mzs").combine_chunks(); itc = t.column("ms2_normalized_intensities").combine_chunks()\n        o = mzc.offsets.to_numpy().astype(np.int64)\n        n = len(o) - 1; k = int(o[-1] - o[0])\n        allmz[p0:p0 + k] = mzc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]\n        allin[p0:p0 + k] = itc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]\n        off[r0 + 1:r0 + n + 1] = p0 + (o[1:] - o[0])\n        prec.append(t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64))\n        add += t.column("adduct").cast(pa.string()).to_pylist()\n        pol.append(np.array(t.column("ionization_mode").cast(pa.string()).to_pylist(), dtype=object) == "positive")\n        ik += t.column("inchikey14").cast(pa.string()).to_pylist()\n        smi += t.column("normalized_smiles").cast(pa.string()).to_pylist()\n        fo += t.column("molecular_formula").cast(pa.string()).to_pylist()\n        tims.append(np.array([isinstance(x, str) and x.lower().replace("-", "").replace(" ", "") == "timstof"\n                              for x in t.column("instrument_type").cast(pa.string()).to_pylist()]))\n        r0 += n; p0 += k\n        del t, mzc, itc\n    prec = np.concatenate(prec); add = np.asarray(add, dtype=object)\n    pol = np.where(np.concatenate(pol), 1, -1).astype(np.int8)\n    nmp = pv.neutral_mass(prec, add)\n    _fm = {x: formula_mass(x) for x in set(v for v in fo if isinstance(v, str))}\n    nmf = np.array([_fm.get(x, np.nan) if isinstance(x, str) else np.nan for x in fo])\n    nm = np.where(np.isfinite(nmf), nmf, nmp)\n    tims = np.concatenate(tims)\n    log(f"library mass index: formula {np.isfinite(nmf).mean():.1%}, timsTOF rows {tims.mean():.1%}")\n    del fo\n    return dict(off=off, mz=allmz, it=allin, prec=prec, nm=nm, pol=pol, tims=tims,\n                ik=np.asarray(ik, dtype=object), smi=np.asarray(smi, dtype=object))\n\n\n# ---------------------------------------------------------------------------------------------\n# candidate pool = COCONUT (prvsiyan fingerprints) U training structures (fingerprinted here)\n# ---------------------------------------------------------------------------------------------\n_g = {}\n\n\ndef _fp_init():\n    from rdkit import RDLogger\n    from rdkit.Chem import rdFingerprintGenerator\n    RDLogger.DisableLog("rdApp.*")\n    _g["bits"] = np.load(find("fp_bits.npy"))\n    _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)\n    _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)\n    _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)\n\n\ndef fp_and_mass(smi):\n    from rdkit import Chem\n    from rdkit.Chem import MACCSkeys\n    from rdkit.Chem.Descriptors import ExactMolWt\n    if not _g: _fp_init()\n    m = Chem.MolFromSmiles(smi)\n    if m is None: return None\n    try:\n        fp = np.concatenate([_g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),\n                             _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),\n                             _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),\n                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["bits"]]\n        return np.packbits(fp), float(ExactMolWt(m))\n    except Exception:\n        return None\n\n\ndef canon_key(smi):\n    from rdkit import Chem, RDLogger\n    from rdkit.Chem.MolStandardize import rdMolStandardize\n    RDLogger.DisableLog("rdApp.*")\n    if "te" not in _g: _g["te"] = rdMolStandardize.TautomerEnumerator()\n    try:\n        m = Chem.MolFromSmiles(smi)\n        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None\n    except Exception:\n        return None\n\n\ndef build_pool(L, workers):\n    from multiprocessing import Pool as MPool\n    d = os.path.dirname(find("coco_fp.npy"))\n    cm = pickle.load(open(os.path.join(d, "coco_meta.pkl"), "rb"))\n    co_fp = np.load(os.path.join(d, "coco_fp.npy")); co_mass = np.load(os.path.join(d, "coco_mass.npy"))\n    co_keys = np.asarray(cm["keys"], dtype=object); co_smi = np.asarray(cm["smiles"], dtype=object)\n    # ADDED: ChEBI + LIPID MAPS (prvsiyan/chebi-lipidmaps-casmi26), same 6,930-bit packed fingerprints\n    bd = os.path.dirname(find("bio_fp.npy")); bm = pickle.load(open(os.path.join(bd, "bio_meta.pkl"), "rb"))\n    bkeys = np.asarray(bm["keys"], dtype=object); bnew = ~pd.Series(bkeys).isin(set(co_keys)).values\n    co_fp = np.vstack([co_fp, np.load(os.path.join(bd, "bio_fp.npy"))[bnew]])\n    co_mass = np.concatenate([co_mass, np.load(os.path.join(bd, "bio_mass.npy"))[bnew]])\n    co_keys = np.concatenate([co_keys, bkeys[bnew]])\n    co_smi = np.concatenate([co_smi, np.asarray(bm["smiles"], dtype=object)[bnew]])\n    log(f"+ ChEBI/LIPID MAPS {int(bnew.sum()):,} new structures")\n    tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")\n    tr = tr[~tr.ik.isin(set(co_keys))]\n    log(f"COCONUT {len(co_keys):,}; training structures to fingerprint {len(tr):,}")\n    with MPool(workers) as mp:\n        res = mp.map(fp_and_mass, list(tr.smi), chunksize=500)\n    ok = np.array([r is not None for r in res])\n    tr_fp = np.stack([r[0] for r in res if r is not None]); tr_mass = np.array([r[1] for r in res if r is not None])\n    fp = np.vstack([co_fp, tr_fp]); mass = np.concatenate([co_mass, tr_mass])\n    keys = np.concatenate([co_keys, tr.ik.values[ok]]); smis = np.concatenate([co_smi, tr.smi.values[ok]])\n    good = np.isfinite(mass)\n    o = np.argsort(np.where(good, mass, 1e18), kind="mergesort")[: good.sum()]\n    P = dict(fp=fp[o], mass=mass[o], keys=keys[o], smiles=smis[o])\n    P["k2i"] = pd.Series(np.arange(len(o)), index=P["keys"]); P["k2i"] = P["k2i"][~P["k2i"].index.duplicated()]\n    log(f"pool {len(o):,} structures")\n    return P\n\n\ndef pool_fps(idx):\n    return np.unpackbits(np.asarray(POOL["fp"][np.asarray(idx, np.int64)]), axis=1)[:, :6930]\n\n\ndef pool_window(t, ppm):\n    a = np.searchsorted(POOL["mass"], t * (1 - ppm / 1e6), "left")\n    b = np.searchsorted(POOL["mass"], t * (1 + ppm / 1e6), "right")\n    return np.arange(a, b)\n\n\n# ---------------------------------------------------------------------------------------------\n# index: neutral-mass sorted library rows + cleaned representatives (all / positive / negative)\n# ---------------------------------------------------------------------------------------------\ndef _reps(L, rows):\n    npk = np.diff(L["off"])[rows]\n    df = pd.DataFrame({"row": rows, "p": L["row_p"][rows], "npk": npk, "tims": L["tims"][rows]})\n    # ADDED: prefer the test instrument (timsTOF) representative, then the richest spectrum\n    df = df.sort_values(["p", "tims", "npk", "row"], ascending=[True, False, False, True]).drop_duplicates("p")\n    rep = df.row.values; rep_nm = L["nm"][rep]\n    o = np.argsort(rep_nm, kind="mergesort"); rep = rep[o]\n    roff, rmz, rit = pv.clean_store(rep, L["off"], L["mz"], L["it"], CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)\n    return dict(rep=rep, rep_p=L["row_p"][rep], rep_nm=L["nm"][rep], roff=roff, rmz=rmz, rit=rit)\n\n\ndef build_index(L):\n    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)\n    rows = np.where(ok)[0]\n    o = np.argsort(L["nm"][rows], kind="mergesort")\n    I = dict(lib_rows=rows[o], lib_nm=L["nm"][rows[o]])\n    I["all"] = _reps(L, rows)\n    I["pos"] = _reps(L, rows[L["pol"][rows] == 1])\n    I["neg"] = _reps(L, rows[L["pol"][rows] == -1])\n    log(f"index: {len(rows):,} library rows; representatives {len(I[\'all\'][\'rep\']):,}")\n    return I\n\n\n# ---------------------------------------------------------------------------------------------\n# per-molecule channels (identical to e02_channels.molecule_channels)\n# ---------------------------------------------------------------------------------------------\ndef clean(mz, it):\n    return pv._clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),\n                     CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)\n\n\ndef analog_search(R, Q, target):\n    lo = np.searchsorted(R["rep_nm"], target - CFG.ANALOG_WIN, "left")\n    hi = np.searchsorted(R["rep_nm"], target + CFG.ANALOG_WIN, "right")\n    asim = np.zeros(hi - lo, np.float32)\n    if hi > lo and Q:\n        shift = (target - R["rep_nm"][lo:hi]).astype(np.float32)\n        for qm, qp in Q:\n            asim = np.maximum(asim, pv.search_shift_pre(qm, qp, lo, hi, R["roff"], R["rmz"], R["rit"], CFG.MZ_TOL, shift))\n    return R["rep_p"][lo:hi], asim\n\n\ndef top_analogs(p, s):\n    if len(p) == 0: return []\n    df = pd.DataFrame({"p": p, "s": s}).groupby("p", sort=False).s.max().sort_values(ascending=False, kind="mergesort")\n    return list(zip(df.index[:CFG.N_ANALOG].astype(int), df.values[:CFG.N_ANALOG].astype(float)))\n\n\ndef molecule_channels(L, I, specs, precs, pols, target):\n    Q = [clean(a, b) for a, b in specs]\n    keep = [i for i, (a, _) in enumerate(Q) if len(a)]\n    Q = [Q[i] for i in keep]; precs = np.asarray(precs)[keep]; pols = np.asarray(pols)[keep]\n    tol = target * CFG.PPM_WIN / 1e6\n    lo = np.searchsorted(I["lib_nm"], target - tol, "left"); hi = np.searchsorted(I["lib_nm"], target + tol, "right")\n    crow = I["lib_rows"][lo:hi]\n    u, inv = np.unique(L["row_p"][crow], return_inverse=True)\n    M = np.zeros((max(len(Q), 1), len(u)), np.float32); MS = np.zeros_like(M)\n    for j, (qm, qp) in enumerate(Q):\n        if len(crow) == 0: break\n        s = pv.search(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,\n                      CFG.INT_POWER, CFG.ENT_WEIGHT)\n        np.maximum.at(M[j], inv, s)\n        sh = (precs[j] - L["prec"][crow]).astype(np.float32)\n        s2 = pv.search_shift_rows(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,\n                                  CFG.INT_POWER, CFG.ENT_WEIGHT, sh)\n        np.maximum.at(MS[j], inv, s2)\n    cnt = np.bincount(inv, minlength=len(u)).astype(np.float32)\n    z = np.zeros(0, np.float32)\n    lib = (u, M.max(0) if len(u) else z, M.mean(0) if len(u) else z, MS.max(0) if len(u) else z, cnt)\n    p, s = analog_search(I["all"], Q, target)\n    ana = top_analogs(p, s)\n    ps, ss = [], []\n    for sign, name in ((1, "pos"), (-1, "neg")):\n        QQ = [q for q, pl in zip(Q, pols) if pl == sign]\n        if QQ:\n            p, s = analog_search(I[name], QQ, target); ps.append(p); ss.append(s)\n    anapol = top_analogs(np.concatenate(ps), np.concatenate(ss)) if ps else []\n    return lib, ana, anapol\n\n\ndef parse_ce(v):\n    try:\n        a = np.abs(np.atleast_1d(np.asarray(v, float)))\n        return float(a.mean()) if len(a) else 25.0\n    except Exception:\n        return 25.0\n\n\n\n\n# ---------------------------------------------------------------------------------------------\n# ranking features: prvsiyan\'s 31 + our blocks (identical to e05_rank.py BLOCKS)\n# ---------------------------------------------------------------------------------------------\ndef _rel(v):\n    v = np.asarray(v, np.float32)\n    return [v, pv._rank_norm(v).astype(np.float32), v - (v.max() if len(v) else 0.0)]\n\n\ndef _analog_feats(cfp, ana):\n    nc = len(cfp)\n    if not ana: return np.zeros((nc, 5), np.float32)\n    afp = pool_fps([p for p, _ in ana]).astype(np.float32); cf = cfp.astype(np.float32)\n    inter = cf @ afp.T\n    tan = inter / (cf.sum(1)[:, None] + afp.sum(1)[None, :] - inter + 1e-9)\n    w = np.clip(np.array([s for _, s in ana], np.float32), 0, None)\n    ap = (tan * (w ** 3)[None, :]).max(1)\n    ap6 = (tan * (w ** 6)[None, :]).max(1)\n    return np.column_stack(_rel(ap) + [ap6, np.full(nc, w[0])]).astype(np.float32)\n\n\ndef blk_base(r, cfp):\n    ana = r["ana"]\n    afp = pool_fps([p for p, _ in ana]) if ana else None\n    sims = np.array([s for _, s in ana], np.float32)\n    return pv.rank_features(cfp, r["lib"], afp, sims, r["zlog"], r["frag"])\n\n\ndef blk_mass(r, cfp):\n    ppm = (POOL["mass"][r["cand"]] - r["target"]) / r["target"] * 1e6\n    return np.column_stack([ppm, np.abs(ppm), np.abs(ppm + 1.5)]).astype(np.float32)\n\n\ndef blk_libx(r, cfp):\n    cnt = np.log1p(r["lib_cnt"])\n    return np.column_stack(_rel(r["lib_mean"]) + _rel(r["libsh"]) + [cnt, cnt - cnt.max()]).astype(np.float32)\n\n\ndef blk_anapol(r, cfp):\n    return _analog_feats(cfp, r["anapol"])\n\n\ndef blk_fragx(r, cfp):\n    return np.column_stack(_rel(r["frag_ad"]) + _rel(r["frag_mean"]) + [r["frag_pol"]]).astype(np.float32)\n\n\nBLOCKS = {"base": blk_base, "mass": blk_mass, "libx": blk_libx, "anapol": blk_anapol, "fragx": blk_fragx}\n\n\n# ---------------------------------------------------------------------------------------------\n# run\n# ---------------------------------------------------------------------------------------------\ndef compute_channels(L, I, models, te):\n    mols = list(te.groupby("molecule_id", sort=True))\n    recs = []\n    for gi, (mid, sub) in enumerate(mols):\n        specs = []\n        for r in sub.itertuples():\n            a_ = np.asarray(r.ms2_mzs, np.float32); b_ = np.asarray(r.ms2_normalized_intensities, np.float32)\n            k_ = a_ <= float(r.precursor_mz) + 2.0\n            a_, b_ = a_[k_], b_[k_]\n            specs.append((a_, b_ / b_.max() if len(b_) and b_.max() > 0 else b_))\n        nms = pv.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.astype(str).values)\n        nms = nms[np.isfinite(nms)]\n        rec = dict(mid=mid, cand=np.zeros(0, np.int64))\n        if len(nms):\n            target = float(np.median(nms))\n            modes = np.where(sub.ionization_mode.astype(str).values == "positive", 1.0, -1.0)\n            lib, ana, anapol = molecule_channels(L, I, specs, sub.precursor_mz.values.astype(np.float64), modes, target)\n            cand = pool_window(target, CFG.PPM_WIN)\n            if len(cand) == 0: cand = pool_window(target, CFG.PPM_FALLBACK)\n            ces = np.array([parse_ce(v) for v in sub.collision_energy_ev.values], np.float32)\n            import pv_fp\n            zlog = pv_fp.molecule_logits(models, specs, sub.precursor_mz.values.astype(np.float32),\n                                         list(sub.adduct.astype(str).values), list(sub.instrument_type.astype(str).values),\n                                         ces, modes)\n            u, lmax, lmean, lsh, lcnt = lib\n            pos = {int(x): i for i, x in enumerate(u)}\n            ci = np.array([pos.get(int(c), -1) for c in cand], np.int64); has = ci >= 0\n            rec.update(target=target, cand=cand, zlog=None if zlog is None else zlog.astype(np.float32), specs=specs,\n                       mode=float(np.mean(modes)), modes=modes, adducts=list(sub.adduct.astype(str).values),\n                       ana=ana, anapol=anapol)\n            for nm_, arr in (("lib", lmax), ("lib_mean", lmean), ("libsh", lsh), ("lib_cnt", lcnt)):\n                v = np.zeros(len(cand), np.float32); v[has] = arr[ci[has]]; rec[nm_] = v\n        recs.append(rec)\n        if gi % 50 == 0: log(f"  channels {gi}/{len(mols)}")\n    return mols, recs\n\n\ndef compute_frag(recs, workers):\n    from multiprocessing import Pool as MPool\n    uc = np.unique(np.concatenate([r["cand"] for r in recs]))\n    log(f"MetFrag-lite on {len(uc):,} candidates")\n    with MPool(workers) as mp:\n        fr = mp.map(pv.frag_masses_safe, list(POOL["smiles"][uc]), chunksize=16)\n    fmap = dict(zip(uc.tolist(), fr))\n    for r in recs:\n        if not len(r["cand"]): continue\n        peaks = []\n        for a_, b_ in r.pop("specs"):\n            m2, i2 = pv._clean(a_, b_, CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)\n            peaks.append((np.asarray(m2, float), np.asarray(i2, float)))\n        F1 = np.zeros((len(r["cand"]), max(len(peaks), 1)), np.float32); F2 = np.zeros_like(F1); F3 = np.zeros_like(F1)\n        for j, c in enumerate(r["cand"]):\n            fm = fmap[int(c)]\n            for k, (x, y) in enumerate(peaks):\n                F1[j, k] = pv.explain_score(fm, x, y, mode=r["mode"], tol=CFG.MZ_TOL)\n                F2[j, k] = pv.explain_score_adduct(fm, x, y, r["adducts"][k], tol=CFG.MZ_TOL)\n                F3[j, k] = pv.explain_score(fm, x, y, mode=r["modes"][k], tol=CFG.MZ_TOL)\n        r["frag"] = F1.max(1); r["frag_ad"] = F2.max(1); r["frag_mean"] = F3.mean(1); r["frag_pol"] = F3.max(1)\n\n\ndef _hgb(X, Y, W, seed, gbm):\n    from sklearn.ensemble import HistGradientBoostingClassifier\n    m = HistGradientBoostingClassifier(random_state=seed, **gbm); m.fit(X, Y, sample_weight=W)\n    return m\n\n\ndef fit_rankers(rank_train_path):\n    """Our ranker: simulation rows from 7 query sets (S = 0 class-1 sim, 1 = class-2 sim)."""\n    z = np.load(rank_train_path, allow_pickle=True)\n    X, Y, S = z["X"], z["Y"], z["S"]\n    rankers = [_hgb(X, Y, np.where(S == 0, wA, 1.0 - wA), sd, RANK.GBM) for wA in RANK.W_A for sd in RANK.SEEDS]\n    log(f"our ranker: {len(rankers)} GBMs on {X.shape[0]:,} rows x {X.shape[1]} features")\n    return rankers, [str(b) for b in z["blocks"]]\n\n\ndef fit_pv_rankers(pv_rank_path, priors=(0.30, 0.60), seeds=(0, 1, 2, 3)):\n    """prvsiyan\'s ranker, exactly as in his notebook: shipped rank_train.npz (M = 0 class-1 sim)."""\n    z = np.load(pv_rank_path)\n    rankers = [_hgb(z["X"], z["Y"], np.where(z["M"] == 0, w1, 1.0 - w1), sd, CFG.GBM) for w1 in priors for sd in seeds]\n    log(f"prvsiyan ranker: {len(rankers)} GBMs on {z[\'X\'].shape[0]:,} rows x {z[\'X\'].shape[1]} features")\n    return rankers, z["X"].shape[1]\n\n\ndef score_molecules(recs, rankers, blocks, use_fp):\n    out = {}\n    for r in recs:\n        if not len(r["cand"]): continue\n        rr = r if use_fp else dict(r, zlog=None)\n        cfp = pool_fps(r["cand"])\n        Xm = np.hstack([BLOCKS[b](rr, cfp) for b in blocks]).astype(np.float32)\n        out[r["mid"]] = np.mean([m.predict_proba(Xm)[:, 1] for m in rankers], axis=0)\n    return out\n\n\ndef blend_scores(a, b, wa):\n    """Within-molecule rank blend (both inputs are per-candidate probabilities)."""\n    out = {}\n    for mid in a:\n        ra = 1.0 - pv._rank_norm(a[mid]); rb = 1.0 - pv._rank_norm(b[mid])\n        out[mid] = wa * ra + (1.0 - wa) * rb + 1e-6 * a[mid]\n    return out\n\n\ndef make_submissions(mols, recs, score_sets, sample_path, workers, topn=25):\n    """score_sets: {name: {mid: scores}} -> {name: submission DataFrame}; one canonicalisation pass for all."""\n    from multiprocessing import Pool as MPool\n    cand = {r["mid"]: r["cand"] for r in recs if len(r["cand"])}\n    ordered = {n: {mid: cand[mid][np.argsort(-sc, kind="mergesort")][:topn + 15] for mid, sc in S.items()}\n               for n, S in score_sets.items()}\n    allc = [v for o in ordered.values() for v in o.values()]\n    short = np.unique(np.concatenate(allc)) if allc else np.zeros(0, np.int64)\n    with MPool(workers) as mp:\n        ck = mp.map(canon_key, list(POOL["smiles"][short]), chunksize=64)\n    ckey = dict(zip(short.tolist(), ck))\n    samp = pd.read_csv(sample_path)\n    subs = {}\n    for n, o in ordered.items():\n        rows = []\n        for mid, _ in mols:\n            out, seen = [], set()\n            for c in o.get(mid, []):\n                k = ckey.get(int(c)) or POOL["keys"][c]\n                if k in seen: continue\n                seen.add(k); out.append(POOL["smiles"][c])\n                if len(out) == topn: break\n            while len(out) < topn:\n                out.append("CCO")\n            rows.append((mid, ";".join(out[:topn])))\n        sub = samp[["molecule_id"]].merge(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), on="molecule_id", how="left")\n        sub["smiles"] = sub["smiles"].fillna("CCO")\n        assert len(sub) == len(samp) and sub.molecule_id.duplicated().sum() == 0 and sub.smiles.isnull().sum() == 0\n        assert (sub.smiles.str.split(";").map(len) <= topn).all()\n        subs[n] = sub\n    return subs\n\n\ndef main(test_path, train_path, sample_path, our_rank_path, pv_rank_path, fp_model_paths, workers=4, limit=0, w_pv=0.88):\n    global POOL\n    import torch\n    import pv_fp\n    L = load_library(train_path); log(f"library {len(L[\'prec\']):,} spectra")\n    POOL = build_pool(L, workers)\n    L["row_p"] = POOL["k2i"].reindex(L["ik"]).fillna(-1).values.astype(np.int64)\n    del L["ik"], L["smi"]\n    I = build_index(L)\n    dev = "cuda" if torch.cuda.is_available() else "cpu"\n    single, merged, _ = pv_fp.load_fp_models(fp_model_paths, dev)\n    log(f"fp models: {len(single)} single + {len(merged)} merged on {dev}")\n    te = pq.read_table(test_path).to_pandas()\n    if limit:\n        keep = sorted(te.molecule_id.unique())[:limit]; te = te[te.molecule_id.isin(keep)]\n    log(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")\n    mols, recs = compute_channels(L, I, (single, merged, dev), te)\n    del L, I\n    compute_frag(recs, workers)\n    ours, blocks = fit_rankers(our_rank_path)\n    s_ours = score_molecules(recs, ours, blocks, use_fp=False)\n    del ours\n    pvr, nfeat = fit_pv_rankers(pv_rank_path)\n    s_pv = score_molecules(recs, pvr, ["base"], use_fp=True)\n    del pvr\n    subs = make_submissions(mols, recs, {"blend": blend_scores(s_pv, s_ours, w_pv), "pv": s_pv, "ours": s_ours},\n                            sample_path, workers)\n    log("submissions ready")\n    return subs, recs, {"pv": s_pv, "ours": s_ours}\n'}
for _n, _src in ENG_FILES.items():
    open(os.path.join(ENG_DIR, _n), 'w').write(_src)
open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write('"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles: [...], keys: [...]}} (blend order, top 40)."""\nimport os, sys, json, time\nimport numpy as np\nif __name__ == \'__main__\':\n    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)\n    test, train, sample, out = sys.argv[1:5]\n    import casmi_engine as E\n    _orig_find = E.find\n    def _find(n):                                  # v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan\'s 6,930\n        if n == \'fp_bits.npy\':\n            return os.path.join(os.path.dirname(_orig_find(\'coco_fp.npy\')), \'fp_bits.npy\')\n        return _orig_find(n)\n    E.find = _find\n    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200\n    import glob\n    fp_models = sorted(p for p in glob.glob(\'/kaggle/input/**/fp_*.pt\', recursive=True) if \'casmi26-fp-models-v2\' in p)\n    print(\'engine fp models\', fp_models, flush=True)\n    T0 = time.time()\n    subs, recs, S = E.main(test, train, sample, E.find(\'sim_rank_rows_nofp.npz\'), E.find(\'rank_train.npz\'), fp_models,\n                           workers=os.cpu_count(), w_pv=0.88)\n    bl = E.blend_scores(S[\'pv\'], S[\'ours\'], 0.88)\n    res = {}\n    for r in recs:\n        mid = r[\'mid\']\n        if mid not in bl:\n            continue\n        order = np.argsort(-bl[mid], kind=\'mergesort\')[:80]\n        smis, keys, seen = [], [], set()\n        for c in r[\'cand\'][order]:\n            s = E.POOL[\'smiles\'][c]; k = E.canon_key(s)\n            if k is None or k in seen:\n                continue\n            seen.add(k); smis.append(s); keys.append(k)\n            if len(smis) >= 40:\n                break\n        res[str(mid)] = dict(smiles=smis, keys=keys)\n    json.dump(res, open(out, \'w\'))\n    print(\'engine lists\', len(res), f\'{time.time()-T0:.0f}s\', flush=True)\n')
ENG = {}
try:
    _t0 = time.time()
    rr = subprocess.run([sys.executable, os.path.join(ENG_DIR, 'eng_runner.py'), os.path.join(COMP, 'test.parquet'),
                         os.path.join(COMP, 'train.parquet'), os.path.join(COMP, 'sample_submission.csv'),
                         '/kaggle/working/eng_lists.json'], capture_output=True, text=True, timeout=4 * 3600, cwd=ENG_DIR)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    ENG = json.load(open('/kaggle/working/eng_lists.json'))
    print('engine lists', len(ENG), f'{time.time()-_t0:.0f}s')
except Exception as e:
    print('ENGINE FAILED -> v4g lists kept:', repr(e))


# %% [markdown]
## 4. PubChem-only channel with a popularity prior
# 
# This channel searches about 105 million PubChem structures that are not in the main candidate pool. Its score combines the spectrum fit with how well documented a structure is.
# 
# $$\text{score} = z(f\cdot z) + 0.25\,\big(\log(1+\#\text{PubChem substances}) + \log(1+\#\text{PubMed papers})\big)$$
# 
# The 200 most documented structures in the mass window are always added to the shortlist. A gate later decides how many PubChem candidates enter the final list. This prior added +0.004 on the leaderboard.


# %% [code]
V17_PATCH = "\n\n# ---- v17 patch: popularity prior (log1p SID + log1p PMID from PubChem, mapped onto this store by stereo-stripped SMILES) ----\n_POP_DIR = os.environ.get('CASMI_POP_DIR'); _POP_LAM = float(os.environ.get('CASMI_POP_LAM', '0')); _POP_UNION = int(os.environ.get('CASMI_POP_UNION', '0'))\n_init_worker0 = init_worker\n\n\ndef init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):\n    _init_worker0(pc_dir, bits_path, pool_meta_path, code_dir)\n    if _POP_DIR:\n        _W['ls'] = np.load(os.path.join(_POP_DIR, 'pc_lsid.npy'), mmap_mode='r')\n        _W['lp'] = np.load(os.path.join(_POP_DIR, 'pc_lpmid.npy'), mmap_mode='r')\n\n\ndef probe_one(task):\n    mid, target, z = task\n    t0 = time.time()\n    chem = _W['chem']; Chem = _W['Chem']; gen = _W['ecfp4']\n    tol = float(target) * PPM * 1e-6\n    a0 = int(np.searchsorted(_W['mass'], float(target) - tol, 'left'))\n    smis = _window(float(target))\n    n = len(smis)\n    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0, fz_top=None)\n    if n == 0:\n        diag['secs'] = time.time() - t0\n        return mid, [], [], [], diag\n    raw_e = _W['raw_e']; z_e = z[_W['sel_e']].astype(np.float32)\n    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)\n    for i, s in enumerate(smis):\n        m = Chem.MolFromSmiles(s)\n        if m is None:\n            continue\n        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True\n    sc1 = E @ z_e\n    sc1[~ok] = -np.inf\n    n1 = min(PC_N1, int(ok.sum()))\n    diag['n_pass1'] = n1\n    if n1 == 0:\n        diag['secs'] = time.time() - t0\n        return mid, [], [], [], diag\n    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]\n    use_pop = ('ls' in _W) and _POP_LAM > 0\n    pop = (np.asarray(_W['ls'][a0:a0 + n], np.float32) + np.asarray(_W['lp'][a0:a0 + n], np.float32)) if use_pop else np.zeros(n, np.float32)\n    if use_pop and _POP_UNION > 0:\n        pp = np.where(ok, pop, -np.inf); ku = min(_POP_UNION, int(ok.sum()))\n        top1 = np.union1d(top1, np.argpartition(-pp, ku - 1)[:ku])\n    bits = _W['bits']\n    fz, idx = [], []\n    for i in top1:\n        fp = chem.raw_fingerprint(smis[i])\n        if fp is None:\n            continue\n        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))\n    diag['n_pass2'] = len(idx)\n    if not idx:\n        diag['secs'] = time.time() - t0\n        return mid, [], [], [], diag\n    fz = np.asarray(fz, np.float64); idx = np.asarray(idx)\n    keyc = {}\n    def key(j):\n        if j not in keyc:\n            keyc[j] = chem.score_key(smis[idx[j]]); diag['n_keyed'] += 1\n        return keyc[j]\n    for j in np.argsort(-fz, kind='stable'):        # gate quantity = v16's pc_fz[0] (best non-pool structure by raw f.z)\n        k = key(j)\n        if k is not None and k not in _W['pool_keys']:\n            diag['fz_top'] = float(fz[j]); break\n    zf = (fz - fz.mean()) / (fz.std() + 1e-9)\n    sc = zf + _POP_LAM * pop[idx]\n    order = np.argsort(-sc, kind='stable')\n    out, out_s, out_k, out_p, out_sc, seen = [], [], [], [], [], set()\n    for j in order:\n        k = key(j)\n        if k is None or k in seen:\n            continue\n        seen.add(k)\n        if k in _W['pool_keys']:\n            diag['n_in_pool'] += 1\n            continue\n        out.append(smis[idx[j]]); out_s.append(float(sc[j])); out_k.append(k); out_p.append(float(pop[idx[j]])); out_sc.append(float(sc[j]))\n        if len(out) >= K:\n            break\n    diag['n_listed'] = len(out)\n    diag['S'] = float(out_sc[0]) if out_sc else 0.0\n    diag['top_pop'] = float(out_p[0]) if out_p else 0.0\n    diag['secs'] = time.time() - t0\n    return mid, out, out_s, out_k, diag\n"
# PubChem-only channel in its own process (S1 code snapshot + FPNet A+B of casmi26-v3-models) -- identical to v3.6
CORE = '"""P_pubchem probe core (lever 3). Embedded verbatim into the Kaggle notebook (written to /kaggle/working and imported,\nso worker processes can import it) and imported directly by the local smoke test.\n\nPer molecule:\n  1. f.z logits of the A+B FPNet ensemble, exactly like Engine.model_logits: z = 0.5 * (mean per-spectrum logits +\n     mean per-polarity merged-spectrum logits); models averaged inside ModelBank.\n  2. PubChem window: +-10 ppm around the target neutral mass (median over the molecule\'s spectra), no subsampling.\n  3. pass 1 (src/casmi/pubchem.py tier): partial f.z on the ECFP4 block of the selected bits, keep top PC_N1 = 1000.\n  4. pass 2: full selected-bit fingerprints, full f.z, sort descending.\n  5. walk down the f.z order, compute the metric score key (tautomer-canonical InChIKey14); keep a structure only if\n     its key is NOT in the candidate pool (train U COCONUT) and not already listed; stop at 25.\n  An empty list becomes the single \'CCO\' fallback (never padded).\nReading: sum of reciprocal ranks of PubChem-only truths / ~130 public molecules.\n"""\nfrom __future__ import annotations\n\nimport os\nimport time\n\nimport numpy as np\n\nPC_N1 = 5000\nPPM = 10.0\nK = 25\n\n_W = {}\n\n\n# ------------------------------------------------------------------------------------------ model part (main process)\ndef molecule_logits(bank, spectra):\n    """spectra: list of dict(mz, it, mode, adduct, prec, ce, ce_n). Returns z_avg (nbits,) float32 or None."""\n    from casmi import chem, fpnet\n    from casmi.spectra import merge_spectra\n    items_s = []\n    for s in spectra:\n        pm, pi = fpnet.prep_peaks(s[\'mz\'], s[\'it\'], s[\'prec\'])\n        ce_ok = s[\'ce\'] is not None and not np.isnan(s[\'ce\'])\n        items_s.append(dict(mz=pm, it=pi, prec=float(s[\'prec\']), adduct_ix=chem.adduct_index(s[\'adduct\']),\n                            ce=float(s[\'ce\']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,\n                            n_merged=min(int(s.get(\'ce_n\', 1)), 8), mode=float(s[\'mode\'])))\n    items_m = []\n    for md in (1, -1):\n        grp = [s for s in spectra if s[\'mode\'] == md]\n        if not grp:\n            continue\n        mm, ii = merge_spectra([(s[\'mz\'], s[\'it\']) for s in grp])\n        prec = float(np.median([s[\'prec\'] for s in grp]))\n        pm, pi = fpnet.prep_peaks(mm, ii, prec)\n        ces = [s[\'ce\'] for s in grp if s[\'ce\'] is not None and not np.isnan(s[\'ce\'])]\n        adducts = [s[\'adduct\'] for s in grp]\n        ad = max(adducts, key=adducts.count)          # deterministic tie-break (first occurrence)\n        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),\n                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,\n                            n_merged=min(sum(max(1, int(s.get(\'ce_n\', 1))) for s in grp), 8), mode=float(md)))\n    if not items_s:\n        return None\n    zs, _ = bank.logits_el(items_s)\n    zm, _ = bank.logits_el(items_m)\n    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)\n\n\n# ------------------------------------------------------------------------------------------ PubChem part (workers)\ndef init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):\n    import sys\n    if code_dir and code_dir not in sys.path:\n        sys.path.insert(0, code_dir)\n    import pandas as pd\n    from casmi import chem\n    _W[\'chem\'] = chem\n    _W[\'mass\'] = np.load(os.path.join(pc_dir, \'pc_mass.npy\'), mmap_mode=\'r\')\n    _W[\'off\'] = np.load(os.path.join(pc_dir, \'pc_off.npy\'), mmap_mode=\'r\')\n    _W[\'buf\'] = np.load(os.path.join(pc_dir, \'pc_smiles.npy\'), mmap_mode=\'r\')\n    bits = np.load(bits_path)\n    _W[\'bits\'] = bits\n    _W[\'sel_e\'] = np.where(bits < 4096)[0]\n    _W[\'raw_e\'] = bits[_W[\'sel_e\']]\n    _W[\'pool_keys\'] = set(pd.read_parquet(pool_meta_path, columns=[\'key\']).key.values.tolist())\n    r = chem._rdkit()\n    _W[\'ecfp4\'] = r[\'gen\'].GetMorganGenerator(radius=2, fpSize=4096)\n    _W[\'Chem\'] = r[\'Chem\']\n\n\ndef _window(target):\n    tol = target * PPM * 1e-6\n    a = int(np.searchsorted(_W[\'mass\'], target - tol, \'left\')); b = int(np.searchsorted(_W[\'mass\'], target + tol, \'right\'))\n    off = np.asarray(_W[\'off\'][a:b + 1]).astype(np.int64)\n    if b <= a:\n        return []\n    raw = bytes(_W[\'buf\'][off[0]:off[-1]])\n    base = off[0]\n    return [raw[off[i] - base:off[i + 1] - base].decode(\'ascii\') for i in range(b - a)]\n\n\ndef probe_one(task):\n    """task = (molecule_id, target_mass, z (nbits,) float32). Returns (molecule_id, [smiles], [f.z], [score keys], diag dict)."""\n    mid, target, z = task\n    t0 = time.time()\n    chem = _W[\'chem\']; Chem = _W[\'Chem\']; gen = _W[\'ecfp4\']\n    smis = _window(float(target))\n    n = len(smis)\n    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0)\n    if n == 0:\n        diag[\'secs\'] = time.time() - t0\n        return mid, [], [], [], diag\n    raw_e = _W[\'raw_e\']; z_e = z[_W[\'sel_e\']].astype(np.float32)\n    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)\n    for i, s in enumerate(smis):\n        m = Chem.MolFromSmiles(s)\n        if m is None:\n            continue\n        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True\n    sc1 = E @ z_e\n    sc1[~ok] = -np.inf\n    n1 = min(PC_N1, int(ok.sum()))\n    diag[\'n_pass1\'] = n1\n    if n1 == 0:\n        diag[\'secs\'] = time.time() - t0\n        return mid, [], [], [], diag\n    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]\n    bits = _W[\'bits\']\n    fz, idx = [], []\n    for i in top1:\n        fp = chem.raw_fingerprint(smis[i])\n        if fp is None:\n            continue\n        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))\n    diag[\'n_pass2\'] = len(idx)\n    order = np.argsort(-np.asarray(fz), kind=\'stable\')\n    out, out_fz, out_k, seen = [], [], [], set()\n    for o in order:\n        s = smis[idx[o]]\n        k = chem.score_key(s)\n        diag[\'n_keyed\'] += 1\n        if k is None or k in seen:\n            continue\n        seen.add(k)\n        if k in _W[\'pool_keys\']:\n            diag[\'n_in_pool\'] += 1\n            continue\n        out.append(s); out_fz.append(float(fz[o])); out_k.append(k)\n        if len(out) >= K:\n            break\n    diag[\'n_listed\'] = len(out)\n    diag[\'secs\'] = time.time() - t0\n    return mid, out, out_fz, out_k, diag\n'
RUNNER = '"""PubChem-only channel for the final submission (runs in its own process so its casmi package\n(S1 snapshot, FPNet A+B) never collides with the v1 engine\'s casmi package).\n\nPer molecule: z = A+B logits (probe_core2.molecule_logits, identical to the probe that scored 0.037);\nbest_pool_fz = max f.z over the pool window (+-10 ppm, fallback 30 ppm) with the same z (as P1 logits.py);\nPubChem-only list = probe_core2.probe_one (top 25 by f.z, keys not in the pool) with f.z values and score keys.\nWrites a json {molecule_id: {pc, pc_fz, pc_keys, best_pool_fz, target}}.\nusage: python pc_runner.py <v3_code_dir> <v3_models_dir> <pool_dir> <pubchem_dir> <test_parquet> <out_json> <n_workers> [max_mols]\n"""\nimport os, sys, json, time\nimport numpy as np, pandas as pd\n\nif __name__ == \'__main__\':\n    V3CODE, V3MOD, POOL_DIR, PC_DIR, TEST, OUT, NW = sys.argv[1:8]\n    MAXM = int(sys.argv[8]) if len(sys.argv) > 8 else 0\n    NW = int(NW)\n    sys.path.insert(0, V3CODE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n    import torch, glob\n    import probe_core2 as pc\n    from casmi import fpnet, chem\n    T0 = time.time()\n    models = sorted(glob.glob(os.path.join(V3MOD, \'fpnet_*.pt\')))\n    assert len(models) == 2, models\n    dev = \'cuda\' if torch.cuda.is_available() else \'cpu\'\n    bank = fpnet.ModelBank(models, device=dev)\n    bits = np.load(os.path.join(POOL_DIR, \'fp_bits.npy\')); nbits = len(bits)\n    pmass = pd.read_parquet(os.path.join(POOL_DIR, \'pool_meta.parquet\'), columns=[\'mass\']).mass.values.astype(np.float64)\n    assert np.all(np.diff(pmass) >= 0), \'pool not mass-sorted\'\n    pfp = np.load(os.path.join(POOL_DIR, \'pool_fp.npy\'), mmap_mode=\'r\')\n    te = pd.read_parquet(TEST)\n    te[\'nm\'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)\n    tasks, extra = [], {}\n    groups = list(te.groupby(\'molecule_id\', sort=False))\n    if MAXM:\n        groups = groups[:MAXM]\n    for mid, sub in groups:\n        nms = sub.nm.values[np.isfinite(sub.nm.values)]\n        if not len(nms):\n            continue\n        target = float(np.median(nms))\n        spectra = []\n        for r in sub.itertuples():\n            ce = r.collision_energy_ev\n            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)\n            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),\n                                mode=1 if r.ionization_mode == \'positive\' else -1, adduct=r.adduct,\n                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,\n                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))\n        spectra = spectra[:16]\n        try:\n            z = pc.molecule_logits(bank, spectra)\n        except Exception as e:\n            print(\'ERROR logits\', mid, repr(e), flush=True); z = None\n        if z is None:\n            continue\n        tol = target * 10e-6\n        a = np.searchsorted(pmass, target - tol, \'left\'); b = np.searchsorted(pmass, target + tol, \'right\')\n        if b <= a:\n            tol = target * 30e-6\n            a = np.searchsorted(pmass, target - tol, \'left\'); b = np.searchsorted(pmass, target + tol, \'right\')\n        if b > a:\n            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)\n            best = float((F @ z.astype(np.float64)).max())\n        else:\n            best = float(\'nan\')\n        extra[mid] = dict(best_pool_fz=best, target=target)\n        tasks.append((mid, target, z))\n    del bank; torch.cuda.empty_cache()\n    print(f\'logits for {len(tasks)} molecules {time.time()-T0:.0f}s\', flush=True)\n    res = {}\n    import multiprocessing as mp\n    ctx = mp.get_context(\'fork\' if sys.platform != \'win32\' else \'spawn\')\n    with ctx.Pool(NW, initializer=pc.init_worker, initargs=(PC_DIR, os.path.join(POOL_DIR, \'fp_bits.npy\'),\n                                                            os.path.join(POOL_DIR, \'pool_meta.parquet\'), V3CODE)) as P:\n        for k, (mid, smis, fzs, keys, d) in enumerate(P.imap_unordered(pc.probe_one, tasks, chunksize=1)):\n            res[mid] = dict(pc=smis, pc_fz=fzs, pc_keys=keys, fz_top=d.get(\'fz_top\'), S=d.get(\'S\'), top_pop=d.get(\'top_pop\', 0.0), **extra[mid])\n            if (k + 1) % 25 == 0 or k + 1 == len(tasks):\n                print(f\'  pubchem {k+1}/{len(tasks)} {time.time()-T0:.0f}s\', flush=True)\n    json.dump(res, open(OUT, \'w\'))\n    print(\'pc_runner done\', len(res), f\'{time.time()-T0:.0f}s\', flush=True)\n'
open('/kaggle/working/probe_core2.py', 'w').write(CORE + V17_PATCH)
open('/kaggle/working/pc_runner.py', 'w').write(RUNNER)
print('v17 patch appended to probe_core2:', 'def probe_one' in V17_PATCH)
PC = {}
try:
    rr = subprocess.run([sys.executable, '/kaggle/working/pc_runner.py', V3CODE, V3MOD, POOL_DIR, PC_DIR,
                         os.path.join(COMP, 'test.parquet'), '/kaggle/working/pc_lists.json', '4'],
                        capture_output=True, text=True, timeout=4 * 3600)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    PC = json.load(open('/kaggle/working/pc_lists.json'))
except Exception as e:
    print('PUBCHEM CHANNEL FAILED -> base lists only:', repr(e))
print('pubchem lists', len(PC), f'{time.time()-T0:.0f}s')

# %% [markdown]
## 5. Engine 1: Ahmed Berat Özer's v4n
# 
# This step builds the spectral library from `train.parquet` and links the precomputed candidate pool: training structures plus COCONUT, about 711k in total. The pool comes with FPNet fingerprints and fragment tables.


# %% [code]
# v1 engine code (the 0.358 model's engine) + fe_v4; pool symlinks; spectrum cache from train.parquet
sys.path.insert(0, V4CODE)
os.makedirs('work', exist_ok=True)
for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy']:
    src = os.path.join(POOL_DIR, f)
    dst = f'work/{f}'
    if not os.path.exists(dst):
        os.symlink(src, dst)
from casmi.build import build_spec_cache
build_spec_cache(os.path.join(COMP, 'train.parquet'), 'work')
import casmi; print('casmi from', casmi.__file__)
sys.path.insert(1, os.path.join(V4CODE, 'fe_v4'))
print(f'cache built {time.time()-T0:.0f}s')

# %% [code]
# v1 engine exactly as in v3.6 (FPNet A, EngineCfg(generate=True)) wrapped by V1FE (recording hooks + families)
import numpy as np, pandas as pd, torch, lightgbm as lgb
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet, chem
import v1engine
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = Library('work'); P = Pool('work')
tfp = np.load('work/train_fp_sel.npy')
_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m
bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)
print('engine FP bank:', _FULL[0] if _FULL else 'fpnet_0 (A) FALLBACK', flush=True)
E = Engine(L, P, tfp, EngineCfg(generate=True), bank)
V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)
if 'derivation' in MAN['families']:
    n_bad = V.mods['derivation'].self_test()
    print('derivation prior self-test mismatches:', n_bad)
    assert n_bad == 0, 'RDKit build differs from the one the derivation priors were built with'
RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))
RK_FEATS = list(RK['features']); RK_B = [lgb.Booster(model_str=s) for s in RK['boosters']]
def rank_score(X, names):
    pos = {n: i for i, n in enumerate(names)}
    miss = [f for f in RK_FEATS if f not in pos]
    assert not miss, f'ranker features not produced (would be zero-filled): {miss}'
    Xs = np.ascontiguousarray(X[:, [pos[f] for f in RK_FEATS]], dtype=np.float32)
    return np.mean([b.predict(Xs, num_threads=4) for b in RK_B], axis=0)
print('engine ready; ranker features', len(RK_FEATS), 'boosters', len(RK_B), f'{time.time()-T0:.0f}s')

# %% [markdown]
## 6. Main candidate lists + pool popularity re-rank
# 
# For each molecule, v4n scores candidates with FPNet fingerprints, its feature families and a 4-booster LightGBM ranker, and keeps the top 60. The list is then re-ranked by
# 
# $$z(\text{ranker score}) + 0.15\,\big(\log(1+\#\text{substances}) + \log(1+\#\text{papers})\big)$$
# 
# Only pool structures are re-ranked. Generated (novel) candidates keep their positions and form their own simulator groups. This step added +0.010 on the leaderboard.


# %% [code]
# all molecules -> base lists (v3.6 spectra construction and ordering)
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
# Keep ranked overflow for append-only final coverage.
BASE_TAIL_CANDIDATES = {str(k): list(v[0]) for k, v in BASE.items()}


# %% [markdown]
## 7. ICEBERG + GLACIER spectrum simulators
# 
# Both models predict each candidate's spectrum and compare it with the measured one. Candidates with the same molecular formula are then re-ranked by
# 
# $$z(\text{score}) + 1.0\,z(\text{ICEBERG}) + 1.0\,z(\text{GLACIER})$$
# 
# Most wrong answers share the right answer's formula, so this is where isomers get separated.
# 
# - Second-engine candidates join the simulator input, so they are scored too.
# - **GLACIER uses [M+H]+ spectra only.** Restricting it this way added +0.006 on the leaderboard.


# %% [code]
# --- CLAW FIX (b418/b419): stage health is now RECORDED, never silently swallowed ---
# `glob('/kaggle/input/**/x')` WITHOUT recursive=True matches one directory level only, so it
# misses /kaggle/input/datasets/<owner>/<slug>/. That silently disabled ICEBERG+GLACIER:
#   ICE FAILED -> ranker order kept: FileNotFoundError('cp313 rdkit wheel not found')  [b417 log]
#   rdkit-2025.3.6-cp312-...whl is not a supported wheel on this platform               [b417 log]
# recursive=True finds the wheel under BOTH mount layouts (same 2025.3.6 cp313 file either way).
CLAW_ICE_HEALTH = 'ICE NOT RUN'
CLAW_GL_HEALTH = 'GL NOT RUN'
BASE_ICE = False         # b421: base-list ICEBERG fusion OFF (b420 showed it costs 0.009); see header
# ICEBERG post-ranker isomer re-scoring (casmi26-iceberg). Optional: any failure keeps the ranker order.
import gc
ICE_SCORES, ice_stats = {}, dict(molecules=0, changed_top25=0, changed_top1=0)


# --- CLAW FIX (b415c): make ICEBERG/GLACIER runnable on a cp313 Kaggle image ---
# The ICEBERG package bundles rdkit-2025.3.6 built for **cp312**; this image is cp313,
# so pip rejects that wheel and install_site() raises before anything is scored
# (n_mols_scored == 0 for ICEBERG *and* GLACIER). The MANIFEST pins 2025.3.6 because
# "FragmentEngine Kekulize differs in 2026.03", so swapping in the runtime's 2026.03
# rdkit would change the scoring. Instead we install the SAME 2025.3.6 release built
# for cp313, so the pinned version is honoured exactly and RDKIT_REQ passes natively.
#
# install_site() is idempotent via <site>/.ice_site_ok, which it compares against the
# name list of `pkg/wheels`. We pre-build the site and write that exact list, so both
# runners short-circuit and never invoke the failing pip.
def _claw_preseed_site(pkg, site):
    import glob as _g, os as _o, shutil as _s, subprocess as _sp, sys as _sys
    srcs = sorted(_g.glob(_o.path.join(pkg, 'wheels', '*.whl.ice')) +
                  _g.glob(_o.path.join(pkg, 'wheels', '*.whl')))
    # install everything EXCEPT the cp-tag-mismatched rdkit
    keep = [w for w in srcs if not _o.path.basename(w).lower().startswith('rdkit')]
    dropped = [_o.path.basename(w) for w in srcs if w not in keep]
    # ...and add the cp313 build of the SAME rdkit release
    new = sorted(_g.glob('/kaggle/input/**/rdkit-2025.3.6-cp313*.whl', recursive=True))
    if not new:
        new = [w for w in _g.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if 'cp313' in w]

    if not new:
        raise FileNotFoundError('cp313 rdkit wheel not found; dataset not attached?')
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
# v4h: GLACIER third re-scoring term (casmi26-glacier) on the SAME runner input; any GLACIER failure -> v4g (ICE only)
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

# %% [markdown]
## 8. Gated PubChem merge
# 
# PubChem-only candidates enter the main list only when there is no strong library match (`lib_max < 0.9`). When the PubChem fingerprint fit beats the best pool candidate by a wide margin, they take slots 2, 4, 6, 8 and 10. Otherwise they take slots 4, 8, 12, 16 and 20.


# %% [code]
# gated PubChem merge (v3.6, unchanged) -> submission.csv
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
# ---- CLAW PROMOTION RULE (b417) --------------------------------------------------
# The shipped gate only INTERLEAVES PubChem-only structures into fixed slots, so the
# main list always keeps rank 1. When the best PubChem proposal is a strong outlier --
# S = z(f.z over the pass-2 set) + 0.25*pop, carried out of the worker -- AND is well
# documented (pop >= 5), let it lead; everything else keeps its usual slot.
# NOTE: S is the Z-SCORED quantity (median ~6). The raw p['pc_fz'][0] (median -1437)
# must never be compared to S_TAU; that would fire on ~71% of molecules.
# EVIDENCE: b416 (promotion, ICEBERG broken) = 0.417 vs base415 (ICE broken, no
# promotion) 0.374 and b415c (ICE working, no promotion) 0.393. The promotion overcame
# losing ICE, so its marginal value is >= +0.024 (est. ~+0.037 using the notebook's own
# ICE +0.013 / GL +0.006 measurements). b417 isolates it on the WORKING-ICE base.
S_TAU, POP_TAU = 6.0, 5.0


def promote(base, base_keys, pc, pc_keys, slots, n=25):
    """Confident PubChem proposal leads the usual list; every other entry keeps its slot."""
    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)
    if not pc:
        return usual
    top = pc[0]
    return ([top] + [x for x in usual if x != top])[:n]


# self-test of both pure functions (this also runs in the cheap smoke pass)
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

# %% [markdown]
## 9. Two-engine fusion + final simulator re-rank
# 
# The two lists are combined by weighted reciprocal-rank fusion. A candidate's score is $\frac{1}{3 + r_{\text{main}}} + \frac{0.6}{3 + r_{\text{engine 2}}}$, and the top 40 are kept. ICEBERG and GLACIER then re-rank same-formula groups of the fused list one last time, and the top 25 become the submission.


# %% [code]
ALPHA, KRR = 0.6, 3.0
# --- ADAPTIVE SIMULATOR FUSION UPGRADE ---
# In 0.438 (v14):
# CELL19_ICE was False (hard zero for ICE when GLACIER is present).
# While equal fusion (lam_ice=1.0, lam_gl=1.0) dropped score to 0.402,
# zeroing ICE completely discards ICEBERG signal even when GLACIER is uncertain/tied.
# Here we implement confidence-gated adaptive fusion:
# 1. When GLACIER is absent (no [M+H]+): full ICEBERG fallback (lam_ice=1.0)
# 2. When GLACIER is present:
#    - If GLACIER margin is tight (< GL_MARGIN_TAU): gentle ICE blend (lam_ice = ICE_SOFT_LAM)
#    - If GLACIER margin is wide (confident): trust GLACIER alone (lam_ice = 0.0, identical to 0.438)
ICE_SOFT_LAM = 0.20        # Gentle complementary weight, prevents noise domination
GL_MARGIN_TAU = 0.03       # GLACIER discrimination margin threshold for isomers

from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF2
from rdkit import Chem as _Ch2
def _form2(s):
    try:
        return _CMF2(_Ch2.MolFromSmiles(s))
    except Exception:
        return s
FUSE_N = 80   # v13: was 40. GLACIER reorders same-formula groups only inside this window, so deep isomers can now reach top slots
def fuse2(v_smis, e_smis, e_keys, n=FUSE_N):
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

def _gl_margin(g_dict, candidate_keys):
    if not g_dict or not isinstance(g_dict, dict):
        return 0.0
    vals = [g_dict[k] for k in candidate_keys if k in g_dict and g_dict[k] is not None]
    if len(vals) < 2:
        return 0.0
    sv = sorted(vals, reverse=True)
    return float(sv[0] - sv[1])

FUSION_OVERFLOW = {}
if ENG:
    v4 = pd.read_csv("submission.csv")
    changed = n_ice = n_ice_only = n_adaptive_blend = 0; out = []
    for mid, s in zip(v4.molecule_id, v4.smiles):
        vs = [x for x in s.split(';') if x and x != 'CCO']
        e = ENG.get(str(mid))
        if e and e['smiles']:
            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])
            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
            if ice:
                try:
                    _g = gl_of(mid)
                    has_gl = bool(_g)
                    n_ice_only += int(not has_gl)
                    if not has_gl:
                        # GLACIER blind spot: full ICE fallback (exact 0.438 logic)
                        o = ice_fuse.rerank(fsm, fk, fsc, [_form2(x) for x in fsm], ice, lam=ICE_LAM, top_n=len(fsm))
                    else:
                        # GLACIER is present: check confidence margin
                        margin = _gl_margin(_g, fk)
                        if margin < GL_MARGIN_TAU:
                            # Low margin / uncertain -> soft complementary ICE blend
                            n_adaptive_blend += 1
                            o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [ice, _g], [ICE_SOFT_LAM, GL_LAM], top_n=len(fsm))
                        else:
                            # High margin / confident -> pure GLACIER (exact 0.438 logic)
                            o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [{}, _g], [0.0, GL_LAM], top_n=len(fsm))
                    n_ice += int(o[:25] != list(range(min(25, len(o)))))
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    print('fused simulator rerank failed', mid, repr(ex))
            changed += int(bool(vs) and fsm[:1] != vs[:1])
            FUSION_OVERFLOW[str(mid)] = list(fsm[25:])
            vs = fsm[:25]
        out.append((mid, ';'.join(vs[:25]) if vs else 'CCO'))
    fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
    assert len(fs) == len(v4) and fs.molecule_id.is_unique and fs.smiles.notna().all()
    assert fs.smiles.map(lambda s: len(s.split(';'))).max() <= 25
    fs.to_csv("submission.csv", index=False)
    print('Adaptive fusion applied: ICE-only blind fallback =', n_ice_only, ', soft ICE adaptive blend =', n_adaptive_blend)

print('V1 inference complete; applying separately audited final pass next.')


# %% [markdown]
## Final metric audit
# 
# The final pass cannot demote any existing metric-equivalent answer: it retains the first
# occurrence of every valid identity in order, then appends new identities up to 25 candidates.
# Runtime assertions check this for every active molecule. This guarantee applies to the
# same predictions and current metric; it does not predict the score of a fresh inference run.
# 
# Outputs: `submission.csv`, `baseline_submission.csv`, `candidate_audit.csv`,
# `candidate_audit.json`, and stage health in `run_manifest.json`.


# %% [code]
from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()
def canonicalize_smiles_to_ik14(smiles: str) -> str:
    """Canonicalize SMILES to InChIKey14, or None if parsing fails.

    Stereochemistry is removed *before* the tautomer canonicalization, not left to fall out of
    the InChIKey14 truncation afterwards. Truncating to InChIKey14 discards the stereo block, so
    it looks like it should already make stereochemistry free -- but the canonicalization runs
    first, and `TautomerEnumerator` will not move a double bond whose position a neighbouring
    stereocentre constrains. The same skeleton drawn with and without that stereocentre can
    therefore land on two different canonical tautomers, and different tautomers are different
    connectivity, which is the part of the InChIKey14 that survives truncation. Stripping first
    makes the canonical form depend only on the skeleton, which is what the competition promises.

    Both the answer key and the submission go through this function, so the two sides can never
    be treated asymmetrically.
    """
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        canonical_mol = _tautomer_canonicalizer.Canonicalize(mol)
        inchikey = Chem.MolToInchiKey(canonical_mol)
        return inchikey[:14]
    # RDKit's MolSanitizeException family subclasses ValueError; RuntimeError covers the C++
    # layer and TypeError covers Boost.Python argument errors. One participant's malformed
    # SMILES must never take the whole scoring run down, so a failure here is just a non-match.
    except (ValueError, RuntimeError, TypeError):
        return None

def heavy_atom_composition(smiles: str):
    """Heavy-atom element counts (excluding H, ignoring charge) as a hashable key, or None
    if parsing fails.

    Cheap (no tautomer enumeration) -- used to gate the expensive canonicalization. Anything
    that shares an InChIKey14 with an accepted structure has the same heavy-atom skeleton, so
    a heavy-atom-composition mismatch lets us skip canonicalization with no change to any
    score. We deliberately do NOT gate on full molecular formula: protomers / charge variants
    (e.g. a zwitterion C(=O)[O-]...[N+] vs its neutral-acid protomer C(=O)O...[N+]) share an
    InChIKey14 but differ in H count and charge, so a formula gate would wrongly reject them.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))

@lru_cache(maxsize=150000)
def metric_identity(smiles):
    key = canonicalize_smiles_to_ik14(smiles)
    return (key, heavy_atom_composition(smiles)) if key else None

def safe_extend(original, extra, limit=25):
    """Preserve first valid metric identities; fill only unoccupied ranks."""
    out, seen = [], set()
    for smi in list(original) + list(extra):
        smi = str(smi).strip()
        identity = metric_identity(smi)
        if identity is not None and identity not in seen:
            seen.add(identity)
            out.append(smi)
            if len(out) == limit:
                break
    return out

def first_ranks(smiles):
    ranks = {}
    for rank, smi in enumerate(smiles, 1):
        identity = metric_identity(smi)
        if identity is not None:
            ranks.setdefault(identity, rank)
    return ranks

def assert_monotone(original, revised):
    old, new = first_ranks(original), first_ranks(revised)
    assert all(identity in new and new[identity] <= rank for identity, rank in old.items())


# %% [code]
METRIC_SOURCE_SHA256 = 'c61d163f1bb768313fd0863822952f035f343924f37e3f5e4b5883ccef7e6201'
# Final pass: no change to any V1 model, fusion weight, promotion gate or simulator.
import collections
import hashlib
from pathlib import Path
assert rdkit.__version__ == '2026.03.3'
assert n_err == 0, f'Base inference errors: {n_err}'
_before = pd.read_csv('submission.csv')
_before.to_csv('baseline_submission.csv', index=False)
_active = set(te.molecule_id.astype(str))
assert _active <= set(map(str, ENG)), 'Engine 2 did not cover all active molecules'
_stage_meta = {}
for _stage, _path in [('ice', 'ice_work/ice_out.json.meta.json'), ('glacier', 'gl_work/gl_out.json.meta.json')]:
    if Path(_path).is_file():
        _stage_meta[_stage] = json.loads(Path(_path).read_text())
    else:
        _stage_meta[_stage] = {'status': 'missing'}
assert _stage_meta['ice'].get('status') == 'ok', _stage_meta['ice']
if (te.adduct == '[M+H]+').any():
    assert _stage_meta['glacier'].get('status') == 'ok', _stage_meta['glacier']

_dm_smiles = _dm_mass = _dm_order = None
_dm_paths = sorted(glob.glob('/kaggle/input/**/dreams_np_mass.npy', recursive=True))
_dreams_used = 0
_audit_rows, _out = [], []
_targets = te.groupby('molecule_id').nm.median().to_dict()
for _row in _before.itertuples(index=False):
    _mid = str(_row.molecule_id)
    _original = [s.strip() for s in str(_row.smiles).split(';') if s.strip()]
    if _mid not in _active:  # Visible smoke placeholders; hidden rerun processes every molecule.
        _out.append((_row.molecule_id, _row.smiles))
        continue
    _ranks = first_ranks(_original)
    _clean = safe_extend(_original, [])
    _extras = list(FUSION_OVERFLOW.get(_mid, [])) + list(BASE_TAIL_CANDIDATES.get(_mid, []))
    _engine = ENG.get(_mid, {})
    _extras.extend(_engine.get('smiles', []))
    _pc = PC.get(_row.molecule_id, PC.get(_mid, {}))
    _extras.extend(_pc.get('pc', []))
    _revised = safe_extend(_clean, _extras)
    _after_ranked = len(_revised)
    _target = float(_targets.get(_row.molecule_id, _targets.get(_mid, np.nan)))
    if len(_revised) < 25 and np.isfinite(_target) and _target > 0 and _dm_paths:
        if _dm_mass is None:
            _dd = Path(_dm_paths[0]).parent
            _dm_mass = np.load(_dd / 'dreams_np_mass.npy').astype(np.float64)
            _dm_smiles = np.load(_dd / 'dreams_np_smiles.npy', allow_pickle=True)
            assert len(_dm_mass) == len(_dm_smiles)
            _dm_order = np.argsort(_dm_mass, kind='stable')
            _dm_sorted = _dm_mass[_dm_order]
        _lo, _hi = np.searchsorted(_dm_sorted, [_target * (1 - 10e-6), _target * (1 + 10e-6)])
        _ix = _dm_order[_lo:_hi]
        _ix = _ix[np.argsort(np.abs(_dm_mass[_ix] - _target), kind='stable')]
        _mass_candidates = [s.decode() if isinstance(s, bytes) else str(s) for s in _dm_smiles[_ix]]
        _revised = safe_extend(_revised, _mass_candidates)
        _dreams_used += len(_revised) - _after_ranked
    assert _revised, f'No valid candidates for {_mid}'
    assert len(_revised) <= 25 and len(first_ranks(_revised)) == len(_revised)
    assert_monotone(_original, _revised)
    _new_ranks = first_ranks(_revised)
    _audit_rows.append(dict(molecule_id=_mid, original_count=len(_original),
        original_unique=len(_ranks), invalid_count=sum(metric_identity(s) is None for s in _original),
        duplicate_count=sum(metric_identity(s) is not None for s in _original) - len(_ranks),
        revised_count=len(_revised), ranked_added=_after_ranked - len(_clean),
        dreams_added=len(_revised) - _after_ranked,
        promoted_original=sum(_new_ranks[k] < v for k, v in _ranks.items()),
        changed=_original != _revised))
    _out.append((_row.molecule_id, ';'.join(_revised)))
_final = pd.DataFrame(_out, columns=['molecule_id', 'smiles'])
_expected = pd.read_csv(os.path.join(COMP, 'sample_submission.csv')).molecule_id
assert _final.molecule_id.is_unique and set(_final.molecule_id) == set(_expected)
assert _final.smiles.notna().all() and _final.smiles.str.split(';').map(len).between(1, 25).all()
if IS_RERUN:
    assert len(_active) == len(_final), 'A hidden molecule was left unprocessed'
_final.to_csv('submission.csv', index=False)
_audit_df = pd.DataFrame(_audit_rows)
_audit_df.to_csv('candidate_audit.csv', index=False)
_summary = dict(experiment='V1.1 metric-safe tail', rdkit=rdkit.__version__,
    hidden_rerun=IS_RERUN, active_molecules=len(_active), output_rows=len(_final),
    changed_molecules=int(_audit_df.changed.sum()),
    invalid_removed=int(_audit_df.invalid_count.sum()),
    equivalent_duplicates_removed=int(_audit_df.duplicate_count.sum()),
    existing_candidates_promoted=int(_audit_df.promoted_original.sum()),
    ranked_candidates_added=int(_audit_df.ranked_added.sum()),
    dreams_candidates_added=int(_audit_df.dreams_added.sum()),
    monotone_rank_assertions_passed=True,
    metric_source_sha256=METRIC_SOURCE_SHA256,
    baseline_sha256=hashlib.sha256(Path('baseline_submission.csv').read_bytes()).hexdigest(),
    submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
    stages=_stage_meta, runtime_seconds=round(time.time()-T0, 1),
    validation_note='Visible test is a smoke check only and overlaps training; no generalization score claimed.')
Path('candidate_audit.json').write_text(json.dumps(_summary, indent=2, default=str))
print(json.dumps({k:v for k,v in _summary.items() if k != 'stages'}, indent=2))
display(_audit_df)


# %% [code]
# Verify full execution; no changes to models, candidates, ranking or fusion.
assert dev == 'cuda' and torch.cuda.is_available(), 'Source GPU required'
assert rdkit.__version__ == '2026.03.3'
assert n_err == 0
_active_ids = set(te.molecule_id.astype(str))
assert set(map(str, BASE)) == _active_ids
assert set(map(str, ENG)) == _active_ids, 'Incomplete second-engine execution'
assert set(map(str, PC)) == _active_ids, 'Incomplete PubChem execution'
_final_check = pd.read_csv('submission.csv')
_sample_check = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
assert list(_final_check.columns) == ['molecule_id', 'smiles']
assert _final_check.molecule_id.is_unique
assert list(_final_check.molecule_id) == list(_sample_check.molecule_id)
_expected_active = set(_sample_check.molecule_id.astype(str)) if IS_RERUN else set(sorted(_sample_check.molecule_id.astype(str))[:SMOKE_N])
assert _active_ids == _expected_active, 'Unprocessed active molecule'
_processed_check = _final_check[_final_check.molecule_id.astype(str).isin(_active_ids)]
assert _final_check.smiles.notna().all()
_counts_check = _processed_check.smiles.str.split(';').map(len)
assert _counts_check.between(1, 25).all()
for _r in _processed_check.itertuples():
    _smis = _r.smiles.split(';')
    assert all(Chem.MolFromSmiles(_s) is not None for _s in _smis), _r.molecule_id
    assert len(first_ranks(_smis)) == len(_smis), _r.molecule_id
assert _summary['active_molecules'] == len(_active_ids)
assert _summary['output_rows'] == len(_final_check)
if IS_RERUN:
    assert len(_active_ids) == len(_final_check)
assert _summary['monotone_rank_assertions_passed']
_health_check = {}
for _name, _stage in _stage_meta.items():
    assert _stage['status'] == 'ok', (_name, _stage)
    assert _stage['n_mols_scored'] == _stage['n_mols_covered'] > 0, (_name, _stage)
    assert _stage.get('n_pred_fail', 0) == 0 and _stage.get('n_feat_fail', 0) == 0
    assert _stage['device'] == 'cuda' and _stage['rdkit'] == '2025.03.6'
    _allowed_errors = []
    for _err in _stage.get('errors', []):
        _match = re.fullmatch(r'smiles rejected \((inchi round trip|elements \[[^\]]*\]|element outside VALID_ATOM_NUM|> 100 heavy atoms)\): (.+)', _err)
        assert _match is not None, (_name, _err)
        _allowed_errors.append(dict(reason=_match.group(1), smiles=_match.group(2)))
    assert _stage.get('n_bad_smiles', 0) == len(_allowed_errors)
    _health_check[_name] = dict(status='ok', scored=_stage['n_mols_scored'], covered=_stage['n_mols_covered'], unsupported=_allowed_errors)
_acceptance = dict(passed=True, source='samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438', source_version=1,
    rows=len(_final_check), processed=len(_active_ids), hidden_rerun=IS_RERUN, preview_placeholders=len(_final_check)-len(_active_ids), base_errors=n_err,
    engine2_molecules=len(ENG), pubchem_molecules=len(PC),
    candidate_min=int(_counts_check.min()), candidate_max=int(_counts_check.max()),
    rdkit=rdkit.__version__, gpu=torch.cuda.get_device_name(0), stages=_health_check,
    ice_only_fusion_molecules=n_ice_only, fusion_window=FUSE_N,
    runtime_seconds=round(time.time()-T0, 2),
    submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
    visible_data_note='Visible test overlaps training; execution acceptance is not a generalization estimate.')
Path('reproduction_acceptance.json').write_text(json.dumps(_acceptance, indent=2))
print(json.dumps(_acceptance, indent=2))


