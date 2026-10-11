"""Commit and push only this requested reproduction; never stage unrelated work."""
import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path
from kaggle_evidence import ROOT, EVID
PROJECT = ROOT.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--pending', action='store_true')
args = parser.parse_args()
summary_name = 'pending_summary.json' if args.pending else 'final_summary.json'
summary = json.loads((EVID / summary_name).read_text())
assert summary['status'] in ({'PENDING', 'RUNNING'} if args.pending else {'COMPLETE'})
assert args.pending or summary['public_mrr25'] is not None

def git(*args):
    return subprocess.check_output(['git', *args], cwd=PROJECT, text=True).strip()

assert git('branch', '--show-current') == 'main'
files = [PROJECT / 'README.md', ROOT / '.gitignore', ROOT / 'README.md', ROOT / 'RESULTS.md']
files += list(ROOT.glob('*.py')) + list((ROOT / 'release').glob('*'))
files += list((ROOT / 'source').glob('*.ipynb')) + [ROOT / 'source/kernel-metadata.json']
files += [EVID / name for name in ['source_api_metadata.json', 'source_page_evidence.json', 'source_output_manifest.json', 'dependencies.json',
    'release_identity.json', 'publication_state.json', 'launched_api_metadata.json', 'platform_acceptance.json', 'source_preview_comparison.json',
    'submission_result.json', summary_name]]
output_names = ['candidate_audit.csv', 'candidate_audit.json', 'run_manifest.json', 'experiment_audit.csv',
    'experiment_audit.json', 'experiment_audit_factorial.csv', 'experiment_audit_factorial.json',
    'experiment_verification.json', 'e004_verification.json', 'output_manifest.json',
    'ice_work/ice_out.json.meta.json', 'gl_work/gl_out.json.meta.json']
files += [EVID / folder / name for folder in ['source_reference', 'platform_output'] for name in output_names]
files += [EVID / 'platform_output/reproduction_acceptance.json', EVID / 'platform_output/api_metadata.json']
assert all(p.is_file() for p in files), [str(p) for p in files if not p.is_file()]
patterns = [rb'AKIA[0-9A-Z]{16}', rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{40,}',
            rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
for path in files:
    assert path.stat().st_size < 50 * 1024 * 1024
    assert not any(re.search(p, path.read_bytes()) for p in patterns), path.name
paths = [str(path.relative_to(PROJECT)) for path in files]
existing_staged = set(git('diff', '--cached', '--name-only').splitlines())
assert existing_staged <= set(paths), 'Unrelated staged files require reconciliation'
git('add', '--', *paths)
git('diff', '--cached', '--check')
staged = git('diff', '--cached', '--name-only').splitlines()
assert set(staged) <= set(paths)
if staged:
    message = (f"Reproduce Canqiang E006 V1 and submit Kaggle {summary['submission_id']}" if args.pending else f"Record Canqiang E006 Kaggle {summary['public_mrr25']:.3f} official result")
    git('commit', '-m', message)
commit = git('rev-parse', 'HEAD')
git('fetch', 'origin')
assert subprocess.run(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], cwd=PROJECT).returncode == 0, 'Remote diverged; do not force push'
git('push', 'origin', 'HEAD:main')
remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
assert remote == commit
sync = dict(commit=commit, pushed=True, remote_commit=remote, files=len(staged),
    submission_id=summary['submission_id'], public_mrr25=summary['public_mrr25'], phase='submitted' if args.pending else 'scored',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(EVID / 'github_sync.json').write_text(json.dumps(sync, indent=2))
print(json.dumps(sync, indent=2))
