"""Manifests and verified source bytes.

A file source is checked against its declared SHA-256 and a Git source against
its full commit hash, so every reader sees exactly the declared content.
"""

from __future__ import annotations

import json
from pathlib import Path

from .datatypes import Manifest, Source
from .errors import IntegrityError, MimiRAGError
from .git import blob, verify_commit
from .integrity import digest, read_bytes


def load_manifest(path: str | Path) -> Manifest:
    """Load a JSON source manifest; every source needs an `id`, a `kind` and a `version`, and IDs are unique."""
    value: Manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    for source in value['documents']:
        missing = {'id', 'kind', 'version'} - set(source)
        if missing:
            raise MimiRAGError(f'Source {source.get("id", "?")} lacks {", ".join(sorted(missing))}')
    ids = [s['id'] for s in value['documents']]
    if len(ids) != len(set(ids)):
        raise MimiRAGError('Duplicate source IDs')
    return value


def source_path(root: Path, source: Source) -> Path:
    """Where a file source lives; relative paths are resolved against `root`."""
    return (Path(root) / source['path']).resolve()


def source_bytes(root: Path, source: Source) -> bytes:
    """Verified bytes of a source."""
    if source['kind'] == 'git':
        repo = (Path(root) / source['repo']).resolve()
        verify_commit(repo, source['version'], source.get('ref'))
        return blob(repo, source['version'], source['path'])
    data = read_bytes(_existing(root, source))
    if digest(data) != source['sha256']:
        raise IntegrityError(f'Source hash mismatch: {source["id"]}')
    return data


def stat_key(root: Path, source: Source) -> list[int] | None:
    """Cheap change detector for a file source; None for a Git source, which its commit fixes."""
    if source['kind'] == 'git':
        return None
    # ctime is included because tools can restore mtime after rewriting a file.
    info = _existing(root, source).stat()
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev]


def _existing(root: Path, source: Source) -> Path:
    path = source_path(root, source)
    if not path.is_file():
        raise IntegrityError(f'Source file missing: {source["id"]}')
    return path
