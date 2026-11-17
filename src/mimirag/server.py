"""Read-only MCP server over a built index and an optional pinned code repository.

Requires the optional `mcp` dependency: `pip install 'mimirag[mcp]'`. The server never
builds or changes the index; a stale index is reported so the operator can rebuild it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import functools
import json
from typing import Any, Literal, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from . import __version__
from .candidates import CandidateIndex, Embedder
from .code import CodeRepository
from .errors import MimiRAGError
from .index import Index
from .packets import compact_packet
from .sources import load_manifest

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
F = TypeVar('F', bound=Callable[..., Any])


def _refusals(function: F) -> F:
    """Report refusals as tool errors, keeping their actionable message."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except MimiRAGError as error:
            raise ToolError(f'{type(error).__name__}: {error}') from error

    return wrapper  # type: ignore[return-value]


def create_server(
    index: Index,
    *,
    code: CodeRepository | None = None,
    embedder: Embedder | None = None,
    vectors: dict[str, list[float]] | None = None,
) -> MCPServer:
    """Tools over `index`; code tools when `code` is given; hybrid search when vectors are given.

    `vectors` must cover the index's complete eligible inventory (see `CandidateIndex.embed`).
    """
    if (embedder is None) != (vectors is None):
        raise MimiRAGError('Hybrid search needs both an embedder and vectors')
    candidates = CandidateIndex(index)
    identity = candidates.inventory()['identity'] if vectors is not None else None
    server = MCPServer(
        'mimirag',
        version=__version__,
        instructions=(
            'Search returns passages with citation locators. Quote only returned text, cite its '
            'locator, and call resolve to verify a citation against the current source.'
        ),
    )

    @server.tool(annotations=READ_ONLY)
    @_refusals
    def search(
        query: str,
        *,
        limit: int = 10,
        mode: Literal['lexical', 'hybrid'] = 'lexical',
        compact: bool = False,
        source: str | None = None,
        family: str | None = None,
        topic: str | None = None,
        version: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Find passages for a query, with citation locators.

        `limit` is 1..20. `mode='hybrid'` adds dense ranking when the server has vectors; its
        packet holds full passages without ranks, neighbouring context or a query trace.
        Filters match source metadata; `since`/`until` are ISO dates.
        """
        if mode == 'lexical':
            packet = dict(
                index.search(
                    query,
                    limit=limit,
                    source=source,
                    family=family,
                    topic=topic,
                    version=version,
                    since=since,
                    until=until,
                )
            )
            return compact_packet(packet) if compact else packet
        if embedder is None or vectors is None or identity is None:
            raise MimiRAGError('Hybrid search is not configured on this server; use mode="lexical"')
        if not 1 <= limit <= 20:
            raise MimiRAGError('Search limit must be 1..20')
        return candidates.hybrid(
            query,
            embedder=embedder,
            vectors=vectors,
            identity=identity,
            limit=limit,
            compact=compact,
            source=source,
            family=family,
            topic=topic,
            version=version,
            since=since,
            until=until,
        )

    @server.tool(annotations=READ_ONLY)
    @_refusals
    def resolve(locator: dict[str, Any]) -> dict[str, Any]:
        """Verify a search locator against the current source and return the cited passage."""
        return dict(index.resolve(locator))

    if code is not None:

        @server.tool(annotations=READ_ONLY)
        @_refusals
        def code_search(query: str, *, limit: int = 10) -> dict[str, Any]:
            """Files containing `query` as a literal, case-insensitive string, at the pinned commit."""
            assert code is not None
            return dict(code.search(query, limit=limit))

        @server.tool(annotations=READ_ONLY)
        @_refusals
        def code_read(path: str, *, first: int = 1, last: int | None = None) -> dict[str, Any]:
            """Lines `first`..`last` of a file at the pinned commit, within the declared scope."""
            assert code is not None
            return dict(code.read(path, first=first, last=last))

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='mimirag-mcp', description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', required=True, help='Manifest JSON file')
    parser.add_argument('--index', required=True, help='Built index file')
    parser.add_argument('--root', default='.', help='Directory that manifest paths are relative to')
    parser.add_argument(
        '--code-qualification',
        default='Source text at the declared commit; it shows what the code says, not how it behaves when run.',
        help="Statement attached to code evidence when the manifest declares 'code'",
    )
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    index = Index(args.index, manifest, root=args.root)
    try:
        with index._open() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
            admitted = json.loads(row[0]) if row else None  # older schemas refuse at the first query
            _, extractor = index._built_identity(db)
    except MimiRAGError as error:
        parser.error(str(error))
    if admitted is not None or extractor['ocr'] is not None:
        # Serving without the build's admission would return superseded sources; without OCR, PDF citations refuse.
        parser.error(
            'index was built with admission or OCR settings; serve it from Python with '
            'create_server(Index(..., admission=..., ocr=...))'
        )
    code = None
    if 'code' in manifest:
        code = CodeRepository(args.root, manifest['code'], qualification=args.code_qualification)
    create_server(index, code=code).run()


if __name__ == '__main__':
    main()
