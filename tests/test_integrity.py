import os

import pytest

from mimirag import integrity
from mimirag.errors import MimiRAGError


def test_digest_and_canonical_json_are_stable():
    expected = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert integrity.digest(b'abc') == expected
    assert integrity.canonical({'b': 1, 'a': ['é', None]}) == '{"a":["é",null],"b":1}'
    assert integrity.fingerprint({'b': 1}) == integrity.digest(b'{"b":1}')


def test_confined_refuses_traversal_absolute_paths_and_links(tmp_path):
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/a.txt').write_text('a')
    (tmp_path / 'outside.txt').write_text('x')
    os.symlink(tmp_path / 'docs/a.txt', tmp_path / 'docs/link.txt')
    assert integrity.confined(tmp_path / 'docs', 'a.txt') == (tmp_path / 'docs/a.txt').resolve()
    for bad in ['../outside.txt', '/etc/hosts', '', 'link.txt']:
        with pytest.raises(MimiRAGError):
            integrity.confined(tmp_path / 'docs', bad)


def test_read_bytes_enforces_the_size_limit(tmp_path, monkeypatch):
    path = tmp_path / 'big.txt'
    path.write_text('0123456789')
    monkeypatch.setattr(integrity, 'MAX_BYTES', 9)
    with pytest.raises(MimiRAGError, match='exceeds'):
        integrity.read_bytes(path)
