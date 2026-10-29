"""In-memory mirror of the index: the same units and BM25 ranking, without building an index.

Experiments can vary ranking inputs (for example, prepend document titles) and
rank with the FTS5 query and tie order that `Index.search` uses. Rankings match
the index when the nodes are all of its passages in manifest order.
"""

from __future__ import annotations

from collections.abc import Mapping
import sqlite3
from types import TracebackType
from typing import Any

from ..chunking import units
from ..index import fts_query
from ..integrity import fingerprint
from .alignment import Documents


def node_id(doc: str, page: Any, first: int, last: int) -> str:
    """ID of a unit: a hash of its document, page and line span."""
    return fingerprint([doc, page, first, last])


def document_nodes(documents: Documents) -> dict[str, dict[str, Any]]:
    """Units of the extracted documents, by node ID, with raw character offsets into their page."""
    nodes = {}
    for doc, pages in documents.items():
        for page, text in pages.items():
            lines = text.splitlines(keepends=True)
            offsets = [0]
            for line in lines:
                offsets.append(offsets[-1] + len(line))
            for first, last, body in units(text):
                start = offsets[first - 1]
                end = offsets[last - 1] + len(lines[last - 1].rstrip('\r\n'))
                nodes[node_id(doc, page, first, last)] = dict(
                    doc=doc,
                    page=page,
                    start=start,
                    end=end,
                    text=text[start:end],
                    ranking_text=body,
                    lines=[first, last],
                )
    return nodes


class MirrorIndex:
    """FTS5 BM25 over nodes, with ties going to insertion order as in the index.

    Optional `headers` (node ID to text) are prepended to the ranked text only.
    Use it as a context manager, or call `close`.
    """

    def __init__(self, nodes: Mapping[str, Mapping[str, Any]], headers: Mapping[str, str] | None = None) -> None:
        self.ids = list(nodes)
        self.db = sqlite3.connect(':memory:')
        headers = headers or {}

        def ranked_text(n: str) -> str:
            body = nodes[n].get('ranking_text', nodes[n]['text'])
            return '\n'.join(filter(None, [headers.get(n, ''), body]))

        try:
            self.db.execute('CREATE VIRTUAL TABLE passage_fts USING fts5(text)')
            rows = [(rowid, ranked_text(n)) for rowid, n in enumerate(self.ids, 1)]
            self.db.executemany('INSERT INTO passage_fts(rowid, text) VALUES (?, ?)', rows)
        except Exception:
            self.db.close()
            raise

    def rank(self, question: str) -> list[str]:
        """Node IDs matching any word of `question`, best first."""
        rows = self.db.execute(
            'SELECT rowid FROM passage_fts WHERE passage_fts MATCH ? ORDER BY bm25(passage_fts), rowid',
            (fts_query(question),),
        )
        return [self.ids[rowid - 1] for (rowid,) in rows]

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> MirrorIndex:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()
