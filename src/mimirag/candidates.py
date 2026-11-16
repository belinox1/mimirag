"""Integrity checked full-text candidates, independent of extraction identity."""

from __future__ import annotations

from typing import Any, Protocol

from .embeddings import normalize
from .errors import MimiRAGError
from .evaluation.fusion import rrf
from .evaluation.mirror import node_id
from .index import SEARCH, Index, _add_filters, fts_query
from .integrity import fingerprint
from .packets import evidence_packet, passage_item


class Embedder(Protocol):
    def embed(self, text: str, *, query: bool = False) -> list[float]: ...


class CandidateIndex:
    """Companion to Index; selections are bound to a complete eligible inventory."""

    def __init__(self, index: Index):
        self.index = index

    def inventory(self, **filters: Any) -> dict[str, Any]:
        with self.index._open() as db:
            excluded, notices = self.index._check_freshness(db)
            conditions = ['1=1']
            arguments: list[Any] = []
            supported = ('source', 'family', 'topic', 'version', 'since', 'until')
            if set(filters) - set(supported):
                raise MimiRAGError('Unsupported candidate filter')
            _add_filters(conditions, arguments, {key: filters.get(key) for key in supported})
            if excluded:
                conditions.append('s.id NOT IN (' + ','.join('?' for _ in excluded) + ')')
                arguments.extend(sorted(excluded))
            rows = db.execute(
                'SELECT p.*, s.description, s.sha256, s.extraction '
                'FROM passage p JOIN source s ON s.id=p.source_id WHERE ' + ' AND '.join(conditions) + ' ORDER BY p.id',
                arguments,
            ).fetchall()
            items = {}
            for row in rows:
                key = node_id(row['source_id'], row['page'], row['first_line'], row['last_line'])
                if key in items:
                    raise MimiRAGError('Duplicate candidate ID')
                item = passage_item(row)
                item.update(excerpt=row['text'], truncated=False)
                items[key] = item
            identity, _ = self.index._built_identity(db)
            receipt = fingerprint([identity, items])
            return dict(identity=receipt, items=items, notices=notices)

    def lexical(self, query: str, *, limit: int = 100, **filters: Any) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise MimiRAGError('Candidate limit must be 1..100')
        inventory = self.inventory(**filters)
        with self.index._open() as db:
            excluded, _ = self.index._check_freshness(db)
            conditions, arguments = ['passage_fts MATCH ?'], [fts_query(query)]
            supported = ('source', 'family', 'topic', 'version', 'since', 'until')
            _add_filters(conditions, arguments, {key: filters.get(key) for key in supported})
            if excluded:
                conditions.append('s.id NOT IN (' + ','.join('?' for _ in excluded) + ')')
                arguments.extend(sorted(excluded))
            rows = db.execute(SEARCH.format(conditions=' AND '.join(conditions)), [*arguments, limit]).fetchall()
            ranking = [(node_id(r['source_id'], r['page'], r['first_line'], r['last_line']), -r['score']) for r in rows]
        ranking = [(key, score) for key, score in ranking if key in inventory['items']][:limit]
        if self.inventory(**filters)['identity'] != inventory['identity']:
            raise MimiRAGError('Inventory changed during ranking')
        return dict(identity=inventory['identity'], ranking=ranking)

    def render(self, ids: list[str], *, identity: str, **filters: Any) -> dict[str, Any]:
        inventory = self.inventory(**filters)
        if inventory['identity'] != identity or len(set(ids)) != len(ids):
            raise MimiRAGError('Stale or duplicate candidate selection')
        if any(key not in inventory['items'] for key in ids):
            raise MimiRAGError('Unknown or ineligible candidate')
        return dict(evidence_packet('search', [inventory['items'][key] for key in ids], notices=inventory['notices']))

    def dense(
        self, query: list[float], vectors: dict[str, list[float]], *, identity: str, limit: int = 100, **filters: Any
    ) -> dict[str, Any]:
        """Exact cosine over an identity-bound eligible inventory."""
        base = self.inventory()
        inventory = self.inventory(**filters)
        if any(base['items'].get(key) != item for key, item in inventory['items'].items()):
            raise MimiRAGError('Inventory changed during ranking')
        if identity != base['identity'] or set(vectors) != set(base['items']):
            raise MimiRAGError('Vector inventory mismatch')
        if type(limit) is not int or not 1 <= limit <= 100 or not query:
            raise MimiRAGError('Invalid dense request')

        q = normalize(query)
        scores = []
        for key, vector in vectors.items():
            normalized = normalize(vector)
            if len(normalized) != len(q):
                raise MimiRAGError('Invalid vector')
            if key in inventory['items']:
                scores.append((key, sum(a * b for a, b in zip(q, normalized, strict=True))))
        if (
            self.inventory()['identity'] != base['identity']
            or self.inventory(**filters)['identity'] != inventory['identity']
        ):
            raise MimiRAGError('Inventory changed during ranking')
        return dict(identity=inventory['identity'], ranking=sorted(scores, key=lambda x: (-x[1], x[0]))[:limit])

    def embed(self, embedder: Embedder) -> dict[str, list[float]]:
        """Vectors for the complete unfiltered inventory, as `dense` and `hybrid` require."""
        inventory = self.inventory()
        vectors = {key: embedder.embed(item['excerpt']) for key, item in inventory['items'].items()}
        if self.inventory()['identity'] != inventory['identity']:
            raise MimiRAGError('Inventory changed during embedding')
        return vectors

    def hybrid(
        self,
        query: str,
        *,
        embedder: Embedder,
        vectors: dict[str, list[float]],
        identity: str,
        limit: int = 10,
        depth: int = 100,
        constant: float = 60,
        **filters: Any,
    ) -> dict[str, Any]:
        """Reciprocal-rank fusion of the top `depth` lexical and dense candidates.

        `identity` is the unfiltered inventory the vectors were computed for.
        """
        if type(limit) is not int or not 1 <= limit <= depth:
            raise MimiRAGError('Hybrid limit must be 1..depth')
        lexical = self.lexical(query, limit=depth, **filters)
        dense = self.dense(embedder.embed(query, query=True), vectors, identity=identity, limit=depth, **filters)
        if lexical['identity'] != dense['identity']:
            raise MimiRAGError('Inventory changed during ranking')
        fused = rrf([[key for key, _ in lexical['ranking']], [key for key, _ in dense['ranking']]], constant=constant)
        return self.render([key for key, _ in fused[:limit]], identity=dense['identity'], **filters)
