"""Read-only Git access pinned to full commit hashes."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

from . import integrity
from .errors import IntegrityError, MimiRAGError

REGULAR_FILE_MODES = ('100644', '100755')


def git(repo: Path, *args: str, success: tuple[int, ...] = (0,)) -> bytes:
    """Standard output of a Git command run in `repo`; any exit status not in `success` raises MimiRAGError."""
    try:
        result = subprocess.run(['git', '-C', str(repo), *args], check=False, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise MimiRAGError(f'git {args[0]} timed out in {repo}') from error
    if result.returncode not in success:
        detail = result.stderr.decode(errors='replace').strip()
        raise MimiRAGError(f'git {args[0]} failed in {repo}: {detail}')
    return result.stdout


def verify_commit(repo: Path, commit: str, ref: str | None = None) -> None:
    """Check that `commit` is a full hash in `repo` and that `ref`, if given, still points at it."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise MimiRAGError('A full 40-character commit hash is required')
    if _resolve(repo, commit) != commit:
        raise IntegrityError('Commit mismatch')
    if ref and _resolve(repo, ref) != commit:
        raise IntegrityError('Reference no longer points at the declared commit')


def _resolve(repo: Path, name: str) -> str:
    """The commit that `name` names in `repo`; a missing commit or ref is an integrity failure."""
    try:
        return git(repo, 'rev-parse', '--verify', name + '^{commit}').decode().strip()
    except MimiRAGError as error:
        raise IntegrityError(f'{name} is not a commit in {repo}') from error


def blob(repo: Path, commit: str, path: str) -> bytes:
    """Bytes of a regular file at `commit`, size-checked before reading."""
    integrity.confined(repo, path)
    # Symbolic links and submodules stored in Git are refused like links on disk.
    record = git(repo, 'ls-tree', '-l', commit, '--', path).decode().split()
    if len(record) < 5 or record[0] not in REGULAR_FILE_MODES:
        raise MimiRAGError('Not a regular file at selected revision')
    if int(record[3]) > integrity.MAX_BYTES:
        raise MimiRAGError('Git source exceeds byte limit')
    return git(repo, 'show', f'{commit}:{path}')
