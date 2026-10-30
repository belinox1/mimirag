"""In-memory mirror of the index: the same units and BM25 ranking, without building an index.

Experiments can vary ranking inputs (for example, prepend document titles) and
rank with the FTS5 query and tie order that `Index.search` uses. Rankings match
the index when the nodes are all of its passages in manifest order.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
import re
import sqlite3
from types import TracebackType
from typing import Any

from ..chunking import units
from ..errors import MimiRAGError
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


TOKENIZERS = frozenset({'unicode61', 'porter unicode61'})
"""FTS5 tokenizers a mirror accepts: the index's own, and the same with Porter (English) stemming."""


class MirrorIndex:
    """FTS5 BM25 over nodes, with ties going to insertion order as in the index.

    Optional `headers` (node ID to text) are prepended to the ranked text only.
    A ranking variant can choose another `tokenizer` from `TOKENIZERS`, and drop
    `stopwords` (compared casefolded) from questions before the index's word cap.
    The defaults rank exactly as the index does. Use it as a context manager, or
    call `close`.
    """

    def __init__(
        self,
        nodes: Mapping[str, Mapping[str, Any]],
        headers: Mapping[str, str] | None = None,
        *,
        tokenizer: str = 'unicode61',
        stopwords: Collection[str] = frozenset(),
    ) -> None:
        if tokenizer not in TOKENIZERS:
            raise MimiRAGError(f'Unknown tokenizer: {tokenizer}')
        self.ids = list(nodes)
        self.stopwords = frozenset(word.casefold() for word in stopwords)
        self.db = sqlite3.connect(':memory:')
        headers = headers or {}

        def ranked_text(n: str) -> str:
            body = nodes[n].get('ranking_text', nodes[n]['text'])
            return '\n'.join(filter(None, [headers.get(n, ''), body]))

        try:
            options = '' if tokenizer == 'unicode61' else f", tokenize='{tokenizer}'"
            self.db.execute(f'CREATE VIRTUAL TABLE passage_fts USING fts5(text{options})')
            rows = [(rowid, ranked_text(n)) for rowid, n in enumerate(self.ids, 1)]
            self.db.executemany('INSERT INTO passage_fts(rowid, text) VALUES (?, ?)', rows)
        except Exception:
            self.db.close()
            raise

    def rank(self, question: str) -> list[str]:
        """Node IDs matching any word of `question`, best first."""
        return [node for node, _ in self.rank_scored(question)]

    def rank_scored(self, question: str) -> list[tuple[str, float]]:
        """Node IDs and raw BM25 scores, ascending (smaller is better).

        Candidate depth is the caller's responsibility. Ties retain insertion order.
        """
        if self.stopwords:
            question = ' '.join(w for w in re.findall(r'\w+', question) if w.casefold() not in self.stopwords)
        rows = self.db.execute(
            'SELECT rowid, bm25(passage_fts) FROM passage_fts WHERE passage_fts MATCH ? '
            'ORDER BY bm25(passage_fts), rowid',
            (fts_query(question),),
        )
        return [(self.ids[rowid - 1], score) for rowid, score in rows]

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> MirrorIndex:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()
