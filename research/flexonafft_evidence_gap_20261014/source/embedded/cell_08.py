"""Read-only E004 observer, before the source smoke switch."""


def _e004_preflight(scope):
    import json
    import os
    import platform
    import subprocess
    import pandas as pd
    import rdkit

    slug = 'enveda-CASMI26-molecule-id-mass-spectra'
    comp = os.path.realpath(scope['COMP'])
    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}
    assert platform.python_version_tuple()[:2] in (('3', '12'), ('3', '13')), 'Python 3.12/3.13 required'
    assert rdkit.__version__ == '2026.03.3'
    # Do not initialize a parent CUDA context before the source's Engine2 subprocess.
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    assert 'T4' in gpu, gpu
    ids = sorted(set(pd.read_parquet(os.path.join(comp, 'test.parquet'),
                                    columns=['molecule_id']).molecule_id.astype(str)))
    sample = pd.read_csv(os.path.join(comp, 'sample_submission.csv'), dtype=str)
    assert ids and set(sample.molecule_id) == set(ids)
    result = dict(runtime_input_path=comp, input_ids=ids, python=platform.python_version(),
                  rdkit=rdkit.__version__, gpu=gpu)
    print('E004_PREFLIGHT', json.dumps({k: v for k, v in result.items() if k != 'input_ids'}))
    return result


E004_PREFLIGHT = _e004_preflight(globals())
del _e004_preflight
