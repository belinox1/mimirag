import os

from helpers import run_git
import pytest

from mimirag import git, integrity
from mimirag.errors import IntegrityError, MimiRAGError


def test_verify_commit_needs_a_full_commit_and_an_unmoved_ref(git_repo):
    repo, first = git_repo({'a.txt': 'one\n'})
    run_git(repo, 'tag', 'release', first)
    git.verify_commit(repo, first, 'release')
    with pytest.raises(MimiRAGError, match='full 40-character'):
        git.verify_commit(repo, first[:12])
    with pytest.raises(IntegrityError, match='not a commit'):
        git.verify_commit(repo, '0' * 40)
    with pytest.raises(IntegrityError, match='not a commit'):
        git.verify_commit(repo, first, 'no-such-tag')
    git_repo({'a.txt': 'two\n'}, repo)
    run_git(repo, 'tag', '-f', 'release')
    with pytest.raises(IntegrityError, match='no longer points'):
        git.verify_commit(repo, first, 'release')


def test_blob_reads_regular_files_only(git_repo, monkeypatch):
    repo, commit = git_repo({'docs/a.txt': 'alpha\n'})
    (repo / 'tool.sh').write_text('echo\n')
    (repo / 'tool.sh').chmod(0o755)
    os.symlink('docs/a.txt', repo / 'alias.txt')
    _, commit = git_repo({}, repo)
    (repo / 'alias.txt').unlink()  # leave only the link stored in Git
    assert git.blob(repo, commit, 'docs/a.txt') == b'alpha\n'
    assert run_git(repo, 'ls-tree', commit, 'tool.sh').startswith('100755')
    assert git.blob(repo, commit, 'tool.sh') == b'echo\n'
    for missing in ['alias.txt', 'missing.txt']:
        with pytest.raises(MimiRAGError, match='regular file'):
            git.blob(repo, commit, missing)
    monkeypatch.setattr(integrity, 'MAX_BYTES', 3)
    with pytest.raises(MimiRAGError, match='byte limit'):
        git.blob(repo, commit, 'docs/a.txt')
