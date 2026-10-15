'''v1 engine code (the 0.358 model's engine) + fe_v4; pool symlinks; spectrum cache from train.parquet'''
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
