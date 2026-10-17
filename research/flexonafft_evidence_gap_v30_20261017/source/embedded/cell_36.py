# ---- E2R2: engine-2 rank-2 insert after the final E006 overlay (MODE 'rank2' only) ----
# Idea: antriksharya, public notebook "CASMI26 | LB 0.443 | v7c + engine-2 rank-2 insert" (Apache 2.0); reimplemented.
# Change: candidates are keyed with this notebook's metric_identity (stereo stripped, tautomer-canonical InChIKey-14
# + heavy-atom composition) instead of chem.score_key, which keeps stereo.


def e2r2_make_key(identity, placeholder='CCO'):
    """SMILES -> metric identity, or None for a non-string, a blank string, the placeholder (or any SMILES with the
    placeholder's identity) and an unparseable SMILES. A token whose key is None is never inserted and never matched."""
    ph = identity(placeholder)

    def key(smiles):
        if not isinstance(smiles, str) or not smiles.strip() or smiles.strip() == placeholder:
            return None
        k = identity(smiles)
        return None if k is None or k == ph else k

    return key


def e2r2_row(toks, eng_list, key, limit=25, placeholder='CCO'):
    """Put engine 2's top candidate x2 at rank 2 of one final list (best first). Pure function.

    x2 is the first entry of `eng_list` that is a non-blank string other than the placeholder. It is used only when
    key(x2) is not None; if that entry is unusable the row is left alone (later entries are not tried).
      empty row, or rank 1 is the placeholder  -> unchanged, 'skipped'
      no engine list                           -> unchanged, 'no_eng'
      x2 missing, unparseable or holds ';'     -> unchanged, 'invalid_x2'
      key(x2) == key(rank 1)                   -> unchanged, 'same_as_rank1'
      x2's identity already at rank 2          -> that token stays, deeper twins of it are dropped, 'already_rank2'
      otherwise                                -> every token with key(x2) is dropped, x2 goes to rank 2, the other
                                                  tokens keep their order, the list is cut to `limit`, 'inserted'
    Returns (new_toks, tag, moved_from). moved_from is 'absent' or the old 1-based rank (str) of x2's identity when
    x2 ends at rank 2, else None.
    """
    toks = list(toks)
    if not toks or not isinstance(toks[0], str) or toks[0].strip() in ('', placeholder):
        return toks, 'skipped', None
    if isinstance(eng_list, str):
        eng_list = [eng_list]
    if not eng_list:
        return toks, 'no_eng', None
    x2 = next((s.strip() for s in eng_list if isinstance(s, str) and s.strip() not in ('', placeholder)), None)
    kx = key(x2) if x2 is not None and not any(c in x2 for c in ';\r\n') else None
    if kx is None:
        return toks, 'invalid_x2', None
    keys = [key(t) for t in toks]
    if kx == keys[0]:
        return toks, 'same_as_rank1', None
    hits = [i for i, k in enumerate(keys) if k == kx]
    if hits and hits[0] == 1:
        return [t for i, t in enumerate(toks) if i == 1 or keys[i] != kx][:limit], 'already_rank2', '2'
    rest = [t for t, k in zip(toks, keys) if k != kx]
    return (rest[:1] + [x2] + rest[1:])[:limit], 'inserted', (str(hits[0] + 1) if hits else 'absent')


def _e2r2_apply(scope, base_bytes, workdir='.'):
    """Apply e2r2_row to every row of the final submission.csv, check the result, then replace the file."""
    import hashlib
    import json
    import os
    from collections import Counter
    from pathlib import Path
    import pandas as pd

    root = Path(workdir)
    sub_path, tmp_path = root / 'submission.csv', root / 'submission_e2r2.tmp.csv'
    (root / 'submission_base_e2r2.csv').write_bytes(base_bytes)
    base = pd.read_csv(root / 'submission_base_e2r2.csv', dtype=str, keep_default_na=False)
    sample = pd.read_csv(os.path.join(scope['COMP'], 'sample_submission.csv'), dtype=str, keep_default_na=False)
    key = e2r2_make_key(scope['metric_identity'])
    eng = scope['ENG']
    assert isinstance(eng, dict), type(eng)
    active = set(scope['te'].molecule_id.astype(str))
    strict = not bool(scope.get('IS_RERUN', True))

    def engine_list(mid):
        e = eng.get(mid)
        if e is None and mid.strip().lstrip('-').isdigit():
            e = eng.get(int(mid))
        if isinstance(e, dict):
            return e.get('smiles') or []
        return list(e) if isinstance(e, (list, tuple)) else []

    def n_dup(ts):
        ks = [k for k in map(key, ts) if k is not None]
        return len(ks) - len(set(ks))

    def check(frame):
        assert list(frame.columns) == ['molecule_id', 'smiles'], list(frame.columns)
        assert list(frame.molecule_id) == list(base.molecule_id)
        assert len(frame) == len(sample) and frame.molecule_id.is_unique
        assert set(frame.molecule_id) == set(sample.molecule_id)
        assert frame.smiles.notna().all() and (frame.smiles.str.strip() != '').all()
        assert frame.smiles.map(lambda v: 1 <= len(v.split(';')) <= 25).all()

    check(base)
    tags, moved, rows, examples = Counter(), Counter(), [], []
    eligible_active = tail_dropped = twins_dropped = 0
    for mid, s in zip(base.molecule_id, base.smiles):
        old = s.split(';')
        new, tag, moved_from = e2r2_row(old, engine_list(mid), key)
        tags[tag] += 1
        eligible_active += int(tag != 'skipped' and mid in active)
        if moved_from is not None:
            moved[moved_from] += 1
        if new == old:
            rows.append((mid, s))
            continue
        # changed row: rank 1 kept, x2's identity exactly once (rank 2), other tokens in order, no new duplicates
        kx = key(new[1])
        n_tw = sum(key(t) == kx for t in old)
        assert tag in ('inserted', 'already_rank2') and mid in active, (mid, tag)
        assert new[0] == old[0] and kx is not None and 2 <= len(new) <= 25 and all(t.strip() for t in new), mid
        assert sum(key(t) == kx for t in new) == 1, mid
        assert [t for t in new if key(t) != kx] == [t for t in old if key(t) != kx][:len(new) - 1], mid
        assert n_dup(new) <= n_dup(old), mid
        twins_dropped += max(0, n_tw - 1)
        tail_dropped += len(old) - n_tw + 1 - len(new)
        if len(examples) < 3:
            examples.append(dict(molecule_id=mid, tag=tag, moved_from=moved_from, before=old[:3], after=new[:3]))
        rows.append((mid, ';'.join(new)))
    result = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    check(result)
    if strict:  # commit run: every processed molecule must reach the overlay with an engine-2 list
        assert eligible_active == len(active) and tags['no_eng'] == 0, (eligible_active, len(active), dict(tags))
    changed = int(sum(a != b for a, b in zip(result.smiles, base.smiles)))
    stats = dict(mode='rank2', key='metric_identity', is_rerun=not strict, rows=len(base),
                 eligible=len(base) - tags['skipped'], inserted=tags['inserted'],
                 moved_from={k: moved[k] for k in sorted(moved, key=lambda k: -1 if k == 'absent' else int(k))},
                 same_as_rank1=tags['same_as_rank1'], no_eng=tags['no_eng'], invalid_x2=tags['invalid_x2'],
                 already_rank2=tags['already_rank2'], skipped=tags['skipped'], changed_rows=changed,
                 tail_dropped=tail_dropped, extra_twins_dropped=twins_dropped,
                 base_sha256=hashlib.sha256(base_bytes).hexdigest())
    new_bytes = base_bytes
    if changed:
        result.to_csv(tmp_path, index=False)
        back = pd.read_csv(tmp_path, dtype=str, keep_default_na=False)
        check(back)
        assert list(back.smiles) == list(result.smiles)
        new_bytes = tmp_path.read_bytes()
    stats['submission_sha256'] = hashlib.sha256(new_bytes).hexdigest()
    (root / 'e2r2_stats.json').write_text(json.dumps(dict(stats, status='applied', examples=examples), indent=1) + '\n')
    if changed:
        os.replace(tmp_path, sub_path)
    print('E2R2_APPLIED', json.dumps(stats), flush=True)
    for ex in examples:
        print('E2R2_EXAMPLE', json.dumps(ex), flush=True)
    return stats


import json as _e2r2_json
import os as _e2r2_os
import sys as _e2r2_sys
import traceback as _e2r2_tb
_E2R2_STRICT = not bool(globals().get('IS_RERUN', True))  # commit run: fail loudly; scoring rerun: keep the E006 file
_e2r2_base_bytes = None
E2R2_STATS = None
try:
    with open('submission.csv', 'rb') as _e2r2_f:  # the final E006 file (cwd is /kaggle/working)
        _e2r2_base_bytes = _e2r2_f.read()
    print('E2R2 reads', _e2r2_os.path.abspath('submission.csv'), len(_e2r2_base_bytes), 'bytes', flush=True)
    E2R2_STATS = _e2r2_apply(globals(), _e2r2_base_bytes)
except Exception as _e2r2_exc:
    if _E2R2_STRICT:
        raise
    _e2r2_state = 'not read'
    if _e2r2_base_bytes is not None:
        try:
            _e2r2_now = None
            if _e2r2_os.path.isfile('submission.csv'):
                with open('submission.csv', 'rb') as _e2r2_f:
                    _e2r2_now = _e2r2_f.read()
            if _e2r2_now != _e2r2_base_bytes:
                with open('submission.csv', 'wb') as _e2r2_f:
                    _e2r2_f.write(_e2r2_base_bytes)
            _e2r2_state = 'untouched' if _e2r2_now == _e2r2_base_bytes else 'restored from the E006 bytes'
        except Exception as _e2r2_exc2:
            _e2r2_state = f'RESTORE FAILED {_e2r2_exc2!r}'
    try:
        if _e2r2_os.path.exists('submission_e2r2.tmp.csv'):
            _e2r2_os.remove('submission_e2r2.tmp.csv')
        with open('e2r2_stats.json', 'w') as _e2r2_f:
            _e2r2_f.write(_e2r2_json.dumps(dict(status='failed', submission=_e2r2_state,
                                                error=f'{type(_e2r2_exc).__name__}: {_e2r2_exc}'[:500])) + '\n')
    except Exception:
        pass
    try:
        print('!' * 100, flush=True)
        print(f'E2R2_FAILED {type(_e2r2_exc).__name__}: {_e2r2_exc} | submission.csv {_e2r2_state} '
              '| the E006 list is the submission', flush=True)
        _e2r2_tb.print_exc(file=_e2r2_sys.stdout)
        print('!' * 100, flush=True)
    except Exception:
        print('E2R2_FAILED (error report failed) | submission.csv', _e2r2_state, flush=True)
