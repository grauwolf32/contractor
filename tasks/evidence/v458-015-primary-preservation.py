from pathlib import Path
import hashlib
import json
import stat
import subprocess
import sys

snapshot = json.loads(Path(__file__).with_name('v458-015-main-integration.json').read_text())
root = Path(snapshot['checkout'])


def git(*args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def check_primary():
    for item in snapshot['files']:
        path = root / item['path']
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Primary file type changed: ' + item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Primary file content changed: ' + item['path'])
        if stat.S_IMODE(path.stat().st_mode) != item['mode']:
            raise RuntimeError('Primary file mode changed: ' + item['path'])
    if hashlib.sha256(git('diff', '--cached', '--binary')).hexdigest() != snapshot['staged_sha256']:
        raise RuntimeError('Primary staged delta changed')


if sys.argv[1:] == ['--verify-only']:
    check_primary()
    print('Preserved', len(snapshot['files']), 'primary files, modes and staged delta')
    raise SystemExit(0)
expected, target = sys.argv[1:]
if git('symbolic-ref', 'HEAD').decode().strip() != 'refs/heads/main':
    raise RuntimeError('Primary checkout is no longer on main')
if git('rev-parse', 'main').decode().strip() != expected:
    raise RuntimeError('Main advanced before installation; reconcile it first')
subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', expected, target], check=True)
check_primary()
changed = [p.decode() for p in git('diff', '--no-renames', '--name-only', '-z', expected, target).split(b'\0') if p]
protected = {item['path'] for item in snapshot['files']}
overlap = sorted(set(changed) & protected)
if set(overlap) - {'tasks/BACKLOG.md'}:
    raise RuntimeError('Merge now overlaps other primary work: ' + repr(overlap))
clean = [path for path in changed if path not in protected]
if clean:
    subprocess.run(['git', '-C', str(root), 'diff', '--quiet', expected, '--', *clean], check=True)
    tracked = set(git('ls-files', '-z').decode().split('\0'))
    for name in clean:
        if name not in tracked and (root / name).exists():
            raise RuntimeError('Untracked primary path would be overwritten: ' + name)
    subprocess.run(['git', '-C', str(root), 'restore', '--source=' + target, '--staged', '--worktree', '--', *clean], check=True)
if overlap:
    # Advance only the committed base of the user's generated backlog.
    # Its working file stays byte-identical to the saved primary snapshot.
    subprocess.run(['git', '-C', str(root), 'restore', '--source=' + target, '--staged', '--', *overlap], check=True)
subprocess.run(['git', '-C', str(root), 'update-ref', '-m', 'merge: land verified material category priority', 'refs/heads/main', target, expected], check=True)
check_primary()
print('Main advanced to', target, '; preserved', len(snapshot['files']), 'primary files and staged delta')
