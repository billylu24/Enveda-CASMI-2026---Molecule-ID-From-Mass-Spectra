'''CASMI26 v4n = v4i + ICE_LAM = GL_LAM = 1.0 (v4l, LB 0.380) + engine FP bank fpnet_full1 (v4m, LB 0.378)'''
'''CASMI26 v4i = v4h + PubChem-only channel pass-1 depth N1 1000 -> 5000 (recall; gate/slots unchanged)'''
'''CASMI26 v4h = v4g + GLACIER third re-scoring term: z(ranker) + 0.5 z(ICEBERG) + 0.5 z(GLACIER) (casmi26-glacier)'''
'''casmi26 v28 = v17 + pool popularity re-rank of the base list (mu 0.15, mode A); v17 = Ahmed v4g + popularity prior in the PubChem list (see make_v17.py)'''
'''CASMI26 v4f = v4b (FPNet A engine bank, LB 0.354) + ICEBERG post-ranker isomer re-scoring = the v3.6 model (v1 engine + FPNet A + moderate generation + gated PubChem channel, LB 0.358)'''
'''+ fe_v4 feature families computed on the SAME v1 engine run (derivation evidence / class-3 priors, fragmentation 2.0,'''
'''analog-structure relations, DreamsFP views) + a ranker retrained on the v1-engine simulation sim1x with them.'''
'''Built by work/analysis/fe_v4/build_v4v1.py -- edit there.'''
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

'''Compatibility repair: select the simulator RDKit 2025.3.6 wheel by interpreter and platform tags, including bundled .whl.ice files. Fail before inference when no compatible build exists; preserve the main RDKit 2026.03.3 environment.'''

def simulator_rdkit_wheel(pkg):
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
    compatible = set(sys_tags())
    bundled = sorted(glob.glob(os.path.join(pkg, 'wheels', 'rdkit-2025.3.6-*.whl*')))
    external = sorted(glob.glob('/kaggle/input/**/rdkit-2025.3.6-*.whl', recursive=True))
    for path in bundled + external:
        name = os.path.basename(path).removesuffix('.ice')
        _, version, _, tags = parse_wheel_filename(name)
        if str(version) == '2025.3.6' and compatible.intersection(tags):
            return path
    raise FileNotFoundError(
        f'RDKit 2025.3.6 wheel compatible with Python {sys.version_info.major}.{sys.version_info.minor} '
        f'and this platform is required for ICEBERG/GLACIER; candidates: {bundled + external}')

_simulator_pkg = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
print('Simulator RDKit wheel:', simulator_rdkit_wheel(_simulator_pkg))
