"""Hashing, canonical JSON and confined, size-bounded file reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import MimiRAGError

MAX_BYTES = 32 * 1024 * 1024
"""Largest source file, in bytes, that MimiRAG reads."""


def digest(data: bytes) -> str:
    """Hex SHA-256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    """Stable JSON text for hashing and storage: sorted keys, no spaces, Unicode kept."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    """SHA-256 of the canonical JSON of `value`."""
    return digest(canonical(value).encode())


def confined(root: str | Path, relative: str) -> Path:
    """Resolve a relative POSIX path inside `root`, refusing traversal and symbolic links."""
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise MimiRAGError('Expected a relative path without traversal')
    base = Path(root).resolve()
    candidate = base.joinpath(*path.parts)
    # A link is refused even if its target lies inside the root.
    inside = [p for p in (candidate, *candidate.parents) if p != base and base in p.parents]
    if any(p.is_symlink() for p in inside):
        raise MimiRAGError('Symbolic links are not source files')
    result = candidate.resolve()
    if not result.is_relative_to(base):
        raise MimiRAGError('Path escapes its source root')
    return result


def read_bytes(path: Path) -> bytes:
    """The bytes of a file of at most MAX_BYTES."""
    if path.stat().st_size > MAX_BYTES:
        raise MimiRAGError(f'Source exceeds {MAX_BYTES} bytes: {path}')
    return path.read_bytes()
