import os

from helpers import source
import pytest

from mimirag.errors import IntegrityError, MimiRAGError
from mimirag.sources import load_manifest, source_bytes, stat_key


def test_manifest_needs_versioned_sources_with_unique_ids(tmp_path):
    path = tmp_path / 'manifest.json'
    entry = '{"id": "a", "kind": "text", "version": "1"}'
    path.write_text(f'{{"documents": [{entry}, {entry}]}}')
    with pytest.raises(MimiRAGError, match='Duplicate'):
        load_manifest(path)
    path.write_text('{"documents": [{"id": "a", "kind": "text"}]}')
    with pytest.raises(MimiRAGError, match='a lacks version'):
        load_manifest(path)


def test_source_bytes_checks_file_hashes_and_git_commits(tmp_path, git_repo):
    (tmp_path / 'a.txt').write_text('alpha')
    entry = source(tmp_path, 'a.txt')
    assert source_bytes(tmp_path, entry) == b'alpha'
    (tmp_path / 'a.txt').write_text('beta')
    with pytest.raises(IntegrityError, match='hash mismatch'):
        source_bytes(tmp_path, entry)
    repo, commit = git_repo({'notes.md': 'note\n'})
    entry = dict(id='n', kind='git', repo=repo.name, path='notes.md', version=commit)
    assert source_bytes(tmp_path, entry) == b'note\n'
    assert stat_key(tmp_path, entry) is None


def test_stat_key_changes_when_a_file_is_touched(tmp_path):
    path = tmp_path / 'a.txt'
    path.write_text('alpha')
    entry = source(tmp_path, 'a.txt')
    before = stat_key(tmp_path, entry)
    os.utime(path, ns=(1, 1))
    assert stat_key(tmp_path, entry) != before
