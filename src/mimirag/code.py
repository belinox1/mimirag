"""Literal search and bounded line reads in a Git repository at one declared commit."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from .datatypes import CodeLocator, CodeSpec, Packet
from .errors import MimiRAGError
from .git import blob, git, verify_commit
from .integrity import digest
from .packets import evidence_packet, excerpt

# File types searched: source code, and text files such as documentation and configuration.
SEARCHED_SUFFIXES = frozenset(
    {'.h', '.hpp', '.cpp', '.cc', '.c', '.py', '.sql', '.rst', '.md', '.toml', '.json', '.yaml', '.yml', '.txt'}
)
MAX_RESULTS = 20
MAX_QUERY = 256
MAX_LINES = 120
READ_WINDOW = 60
LINES_BEFORE_MATCH = 5
LINES_AFTER_MATCH = 15
ORDER = 'path order, first match per file; not relevance ranked'
FRESHNESS = 'verified at read time'


def in_scope(path: str, prefixes: Iterable[str]) -> bool:
    """Whether a relative path equals or lies under an allowed prefix, by whole components."""
    parts = PurePosixPath(path)
    if parts.is_absolute() or '..' in parts.parts or not path:
        raise MimiRAGError('Invalid source path')
    return any(path == prefix.rstrip('/') or path.startswith(prefix.rstrip('/') + '/') for prefix in prefixes)


class CodeRepository:
    """A declared code repository, readable only at its commit and within its scope.

    `spec` gives the repository path, the full commit hash as `version`, an
    optional `ref` that must still point at it and `scope`, the readable path
    prefixes. Every evidence item carries `qualification`, the deployment's
    statement of what source text can and cannot show. Both methods accept the
    `version` of a code locator, so a caller can pass it through; any version other
    than the declared commit is refused.
    """

    def __init__(self, root: str | Path, spec: CodeSpec, *, qualification: str) -> None:
        self.spec = spec
        self.repo = (Path(root) / spec['repo']).resolve()
        self.qualification = qualification

    def search(self, query: str, *, limit: int = 10, version: str | None = None) -> Packet:
        """Files whose path or text contains `query` as a literal, case-insensitive string.

        Each matching file gives one item, its first matching line with a window of
        context, in path order. Query and file text are never used as patterns. Git
        preselects the files, so text matches need Git's case folding (`STRASSE` does
        not find `Straße`).
        """
        if not 1 <= limit <= MAX_RESULTS:
            raise MimiRAGError(f'limit must be 1..{MAX_RESULTS}')
        if not query:
            raise MimiRAGError('A nonempty query is required')
        if len(query) > MAX_QUERY:
            raise MimiRAGError(f'Query exceeds {MAX_QUERY} characters')
        commit = self._commit(version)
        needle = query.casefold()
        candidates = {p for p in self._tracked(commit) if needle in p.casefold()} | self._grep(query, commit)
        hits: list[dict[str, Any]] = []
        scanned = 0
        for path in sorted(candidates):
            if PurePosixPath(path).suffix.lower() not in SEARCHED_SUFFIXES or not in_scope(path, self.spec['scope']):
                continue
            try:
                data = blob(self.repo, commit, path)
                lines = data.decode('utf-8').splitlines()
            except (UnicodeError, MimiRAGError):  # not UTF-8 text, a link or too large
                continue
            scanned += 1
            match = next(
                (i for i, line in enumerate(lines, 1) if needle in line.casefold() or needle in path.casefold()), None
            )
            if match is not None:
                first = max(1, match - LINES_BEFORE_MATCH)
                hits.append(self._item(path, data, lines, first, min(len(lines), match + LINES_AFTER_MATCH)))
            if len(hits) > limit:
                break
        return evidence_packet(
            'code',
            hits[:limit],
            query=query,
            version=commit,
            scanned_files=scanned,
            more_results=len(hits) > limit,
            order=ORDER,
        )

    def read(self, path: str, *, first: int = 1, last: int | None = None, version: str | None = None) -> Packet:
        """Lines `first` to `last` (inclusive, at most 120) of a tracked file in scope.

        Without `last`, up to 60 lines from `first` are returned.
        """
        commit = self._commit(version)
        if not in_scope(path, self.spec['scope']) or path not in self._tracked(commit):
            raise MimiRAGError('Path outside allowlisted tracked source files')
        data = blob(self.repo, commit, path)
        lines = _text(path, data).splitlines()
        if last is None:
            last = min(first + READ_WINDOW - 1, len(lines))
        return evidence_packet('code', [self._item(path, data, lines, first, last)], version=commit)

    def _commit(self, version: str | None) -> str:
        """The declared commit, checked; any other requested version is refused."""
        commit = version or self.spec['version']
        if commit != self.spec['version']:
            raise MimiRAGError('Only the declared code version can be read')
        verify_commit(self.repo, commit, self.spec.get('ref'))
        return commit

    def _tracked(self, commit: str) -> list[str]:
        # NUL-separated output gives paths verbatim, without Git's quoting of unusual names.
        listing = git(self.repo, 'ls-tree', '-r', '-z', '--name-only', commit, '--', *self.spec['scope'])
        return listing.decode().split('\0')[:-1]

    def _grep(self, query: str, commit: str) -> set[str]:
        """Paths of text files in scope containing `query`, by Git's fixed-string search."""
        args = ['grep', '-I', '-l', '-z', '-F', '-i', '-e', query, commit, '--', *self.spec['scope']]
        output = git(self.repo, *args, success=(0, 1))  # 1 means no match
        return {name.split(':', 1)[1] for name in output.decode().split('\0')[:-1]}

    def _item(self, path: str, data: bytes, lines: list[str], first: int, last: int) -> dict[str, Any]:
        if not 1 <= first <= last <= len(lines) or last - first >= MAX_LINES:
            raise MimiRAGError(f'Select 1..{MAX_LINES} existing lines')
        locator = CodeLocator(
            tool='code',
            source=path,
            lines=[first, last],
            sha256=digest(data),
            version=self.spec['version'],  # the only commit `_commit` accepts
            repo=self.spec['repo'],
            path=path,
        )
        return dict(
            source=path,
            locator=locator,
            **excerpt('\n'.join(lines[first - 1 : last])),
            qualification=self.qualification,
            freshness=FRESHNESS,
        )


def _text(path: str, data: bytes) -> str:
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise MimiRAGError(f'Not UTF-8 text: {path}') from error
