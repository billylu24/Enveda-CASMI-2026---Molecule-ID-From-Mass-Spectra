"""Push only this reproduction's reviewed files; record pending and scored updates."""
import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
E = ROOT / 'evidence'
parser = argparse.ArgumentParser()
parser.add_argument('--pending', action='store_true')
args = parser.parse_args()
result = json.loads((E / ('submission_result.json' if args.pending else 'final_summary.json')).read_text())
if args.pending:
    assert result['status'] in {'PENDING', 'RUNNING'}
    submission_id, score = result['ref'], None
    phase = 'submitted'
else:
    assert result['status'] == 'COMPLETE' and result['public_mrr25'] is not None
    submission_id, score = result['submission_id'], result['public_mrr25']
    phase = 'scored'

def run(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=PROJECT, text=True).strip()

assert run('branch', '--show-current') == 'main'
assert not run('diff', '--cached', '--name-only'), 'Unrelated staged files require reconciliation'
files = [PROJECT/'README.md', ROOT/'.gitignore', ROOT/'README.md', ROOT/'RESULTS.md', ROOT/'METHODS_ZH.md']
files += list(ROOT.glob('*.py')) + list((ROOT/'release').glob('*'))
files += list((ROOT/'source').glob('*.ipynb')) + [ROOT/'source/kernel-metadata.json']
evidence_names = ['source_api_metadata.json', 'source_output_manifest.json', 'dependencies.json',
    'release_identity.json', 'publication_state.json', 'launched_api_metadata.json',
    'platform_acceptance.json', 'submission_result.json', 'source_preview_comparison.json',
    'source_reference_manifest.json', 'source_preview_rows.csv']
if not args.pending:
    evidence_names.append('final_summary.json')
files += [E/name for name in evidence_names]
output_names = ['reproduction_acceptance.json', 'run_manifest.json', 'candidate_audit.json',
    'candidate_audit.csv', 'baseline_submission.csv', 'api_metadata.json',
    'ice_work/ice_out.json.meta.json', 'gl_work/gl_out.json.meta.json']
files += [E/'platform_output'/name for name in output_names]
assert all(path.is_file() for path in files), [str(path) for path in files if not path.is_file()]
patterns = [rb'AKIA[0-9A-Z]{16}', rb'gh[pousr]_[A-Za-z0-9]{30,}',
    rb'github_pat_[A-Za-z0-9_]{40,}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
for path in files:
    assert path.stat().st_size < 50*1024*1024, path.name
    assert not any(re.search(pattern, path.read_bytes()) for pattern in patterns), f'Credential pattern in {path.name}'
paths = [str(path.relative_to(PROJECT)) for path in files]
run('add', '--', *paths)
run('diff', '--cached', '--check')
staged = run('diff', '--cached', '--name-only').splitlines()
assert set(staged) <= set(paths), 'Unexpected staged changes'
if staged:
    message = (f'Reproduce Samanyu ICEBERG V1 and submit Kaggle {submission_id}' if args.pending
               else f'Record Samanyu ICEBERG Kaggle {score:.3f} official result')
    run('commit', '-m', message)
commit = run('rev-parse', 'HEAD')
state_file = E / 'github_sync.json'
state = json.loads(state_file.read_text()) if state_file.exists() else {'history': []}
state['history'].append(dict(commit=commit, phase=phase, submission_id=submission_id,
    score=score, committed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=staged))
state.update(commit=commit, phase=phase, submission_id=submission_id, score=score, pushed=False)
state_file.write_text(json.dumps(state, indent=2))
run('fetch', 'origin')
assert subprocess.run(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], cwd=PROJECT).returncode == 0, 'Remote diverged; reconcile without force push'
run('push', 'origin', 'HEAD:main')
remote = run('ls-remote', 'origin', 'refs/heads/main').split()[0]
assert remote == commit, 'Remote head mismatch'
state.update(pushed=True, remote_commit=remote, verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
state_file.write_text(json.dumps(state, indent=2))
print(json.dumps(dict(commit=commit, phase=phase, pushed=True, submission_id=submission_id, score=score, files=len(staged)), indent=2))
