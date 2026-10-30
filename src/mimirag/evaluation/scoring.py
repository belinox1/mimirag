"""Retrieval scoring against reference cases.

A case lists required facts (nuggets) and the gold evidence that supports them.
A nugget is covered when retrieved units, within a rank or character budget,
cover every evidence item of one of its alternatives. All functions are pure:
they take the run's extracted documents, its node inventory and ranked node IDs.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
import re
from typing import Any, Literal

from ..errors import MimiRAGError
from .alignment import Documents, Span, align, cited_page, cited_pages, document_key, tokens

SCORER = 'scorer-v1.2'
"""Label of the scoring rules; it changes whenever a score could."""

TEXT_KINDS = frozenset({'text-layer', 'git-blob'})
PAGE_IMAGE = 'page-image'

TOKEN = re.compile(r'\w+|[^\w\s]')
"""Approximate token: a word or a single punctuation mark; not the token of any language model."""

Nodes = Mapping[str, Mapping[str, Any]]
"""Retrievable units by ID, each with `doc`, `page`, raw `start` and `end` offsets and `text`."""
PageImageRule = Literal['strict', 'lenient']
Status = Literal['retrieved', 'chunk-boundary', 'missed', 'unmapped', 'uncovered']


def within_budget(
    ranked: Sequence[str], nodes: Nodes, *, k: int | None = None, characters: int | None = None
) -> list[str]:
    """The ranked nodes kept by a rank cut-off or a character budget (fairer across unit sizes).

    The budget keeps the longest prefix of the ranking that fits; it stops at the first
    node that would overflow it rather than skipping to smaller ones.
    """
    kept: list[str] = []
    used = 0
    for node_id in ranked[:k] if k is not None else ranked:
        if characters is not None:
            used += len(nodes[node_id]['text'])
            if used > characters:
                break
        kept.append(node_id)
    return kept


def packet_cost(ranked: Iterable[str], nodes: Nodes, useful: int) -> dict[str, Any]:
    """Characters and approximate tokens (`TOKEN`) of the ranked units, per useful evidence item.

    `useful` is the caller's count of evidence items the units retrieve (for example,
    vital items in the `retrieved` state). With none, the per-item costs are None.
    """
    texts = [nodes[node_id]['text'] for node_id in ranked]
    characters = sum(len(text) for text in texts)
    tokens = sum(len(TOKEN.findall(text)) for text in texts)
    return dict(
        characters=characters,
        tokens=tokens,
        useful=useful,
        characters_per_item=characters / useful if useful else None,
        tokens_per_item=tokens / useful if useful else None,
    )


def evidence_status(
    evidence: Mapping[str, Any],
    span: Span | None,
    nodes: Nodes,
    retrieved: Collection[str],
    *,
    overlap: float,
    page_image: PageImageRule,
    source_map: Mapping[str, str] | None = None,
    documents: Documents | None = None,
) -> Status:
    """State of one evidence item: retrieved, chunk-boundary, missed, unmapped or uncovered.

    Text evidence is retrieved when the retrieved nodes together cover every
    non-whitespace character of its span, so overlapping or nested units cost
    nothing; partial cover is chunk-boundary. A span that the nodes together do
    not cover is uncovered (missed for Git evidence).

    Page-image evidence (no usable text layer) counts nodes on the cited pages:
    'lenient' accepts any such node, 'strict' needs `overlap` of the quote's words.
    """
    if evidence['verified_against'] == PAGE_IMAGE:
        return _page_image_status(
            evidence,
            nodes,
            retrieved,
            overlap=overlap,
            page_image=page_image,
            source_map=source_map,
            documents=documents,
        )
    if span is None:
        return 'unmapped'
    if documents is None:
        raise MimiRAGError('Text evidence needs the documents it was aligned against')
    doc, page, start, end = span
    text = documents[doc][page]
    needed = {i for i in range(start, end) if not text[i].isspace()}
    covering = [
        n for n, v in nodes.items() if v['doc'] == doc and v['page'] == page and v['start'] < end and start < v['end']
    ]
    if not covering or not needed <= _held(covering, nodes, start, end):
        # Code nodes exist only for the windows code search returned, so an uncovered Git span
        # was missed; document units cover every page, so an uncovered span shows a unit gap.
        return 'missed' if evidence['verified_against'] == 'git-blob' else 'uncovered'
    if needed <= _held([n for n in covering if n in retrieved], nodes, start, end):
        return 'retrieved'
    return 'chunk-boundary' if any(n in retrieved for n in covering) else 'missed'


def _held(node_ids: Iterable[str], nodes: Nodes, start: int, end: int) -> set[int]:
    """Offsets in [start, end) that the given nodes contain."""
    return {i for n in node_ids for i in range(max(nodes[n]['start'], start), min(nodes[n]['end'], end))}


def _page_image_status(
    evidence: Mapping[str, Any],
    nodes: Nodes,
    retrieved: Collection[str],
    *,
    overlap: float,
    page_image: PageImageRule,
    source_map: Mapping[str, str] | None,
    documents: Documents | None,
) -> Status:
    pages, doc = cited_pages(evidence), document_key(evidence, source_map)
    on_page = [n for n in retrieved if nodes[n]['doc'] == doc and nodes[n]['page'] in pages]
    if page_image == 'lenient':
        return 'retrieved' if on_page else 'missed'
    extracted = (documents or {}).get(doc, {})
    missing = [p for p in pages if p not in extracted]
    if not documents or missing:
        raise MimiRAGError(f'Strict page-image scoring needs the cited extracted page {(missing or pages)[0]} of {doc}')
    quote = tokens(evidence['quote'])

    def words(n: str) -> set[str]:
        return tokens(extracted[nodes[n]['page']][nodes[n]['start'] : nodes[n]['end']])

    return 'retrieved' if quote and any(len(quote & words(n)) >= overlap * len(quote) for n in on_page) else 'missed'


def score_case(
    case: Mapping[str, Any],
    support: Mapping[str, Any],
    documents: Documents,
    nodes: Nodes,
    ranked: Sequence[str],
    *,
    style: str,
    overlap: float,
    page_image: PageImageRule = 'strict',
    k: int | None = None,
    characters: int | None = None,
    named_sources: Iterable[str] | None = None,
    source_map: Mapping[str, str] | None = None,
    added_credit: Literal['named', 'unrestricted'] = 'named',
    cited_page_only: bool = False,
) -> dict[str, Any]:
    """Share of the case's required nuggets covered for one question style, with evidence states.

    `support` maps each nugget to its styles, a coverage-gap flag and alternative
    evidence sets; a nugget is covered when every item of one alternative is
    retrieved, and `any_evidence` relaxes this to any single item. An alternative
    marked `added` (added to a key after it was first fixed) counts only when all its
    sources are in `named_sources`, unless unrestricted credit is chosen for the
    blind style. With `cited_page_only`, a match off the cited pages counts as unmapped.
    """
    if page_image not in {'strict', 'lenient'}:
        raise MimiRAGError(f'Unknown page-image rule: {page_image}')
    if added_credit not in {'named', 'unrestricted'} or (added_credit == 'unrestricted' and style != 'blind'):
        raise MimiRAGError('Unrestricted added-alternative credit is only valid for blind style')
    evidence = {e['id']: e for e in case['evidence']}
    _check_scorable(evidence.values())
    spans = {i: align(e, documents, source_map) for i, e in evidence.items() if e['verified_against'] in TEXT_KINDS}
    if cited_page_only:
        spans = {i: _on_cited_page(span, evidence[i]) for i, span in spans.items()}
    retrieved = set(within_budget(ranked, nodes, k=k, characters=characters))
    states = {
        i: evidence_status(
            e,
            spans.get(i),
            nodes,
            retrieved,
            overlap=overlap,
            page_image=page_image,
            source_map=source_map,
            documents=documents,
        )
        for i, e in evidence.items()
    }
    off_page = sorted(i for i, span in spans.items() if span and _off_cited_pages(span, evidence[i]))
    named = set(named_sources or ())
    covered: dict[str, bool] = {}
    any_covered: dict[str, bool] = {}
    for requirement, value in support.items():
        if style not in value['styles'] or value['coverage_gap']:
            continue
        alternatives = _creditable(value['alternatives'], evidence, added_credit, named)
        covered[requirement] = any(all(states[i] == 'retrieved' for i in ids) for ids in alternatives)
        any_covered[requirement] = any(states[i] == 'retrieved' for ids in alternatives for i in ids)
    return dict(
        coverage=_share(covered),
        any_evidence=_share(any_covered),
        requirements=list(covered),
        covered=covered,
        states=states,
        off_page=off_page,
    )


def _check_scorable(evidence: Iterable[Mapping[str, Any]]) -> None:
    for e in evidence:
        if e.get('verified_against') not in TEXT_KINDS | {PAGE_IMAGE}:
            raise MimiRAGError(
                f'Unsupported evidence kind in {e.get("id")}: only text-layer, git-blob and '
                'page-image evidence can be scored'
            )
        if e['verified_against'] == PAGE_IMAGE and cited_page(e) is None:
            raise MimiRAGError(f'Page-image evidence {e["id"]} has no parseable page locator')


def _on_cited_page(span: Span | None, evidence: Mapping[str, Any]) -> Span | None:
    """The span, or None when it lies off the pages the evidence cites."""
    return None if span and _off_cited_pages(span, evidence) else span


def _off_cited_pages(span: Span, evidence: Mapping[str, Any]) -> bool:
    pages = cited_pages(evidence)
    return bool(pages) and span[1] not in pages


def _creditable(
    alternatives: Iterable[Any], evidence: Mapping[str, Mapping[str, Any]], added_credit: str, named: set[str]
) -> list[list[str]]:
    """Evidence ID lists of the alternatives that may earn credit."""
    result = []
    for alternative in alternatives:
        ids = alternative['evidence'] if isinstance(alternative, dict) else alternative
        added = isinstance(alternative, dict) and alternative.get('added')
        if added and added_credit != 'unrestricted' and not {evidence[i]['source'] for i in ids} <= named:
            continue
        result.append(ids)
    return result


def _share(values: Mapping[str, bool]) -> float | None:
    return sum(values.values()) / len(values) if values else None


def reciprocal_rank(
    case: Mapping[str, Any],
    documents: Documents,
    nodes: Nodes,
    ranked: Sequence[str],
    *,
    evidence_ids: Collection[str] | None = None,
    source_map: Mapping[str, str] | None = None,
) -> float | None:
    """1/rank of the first node overlapping an aligned gold span.

    0.0 if no node overlaps; None when no evidence aligns.

    Only text-layer and Git evidence count, optionally only the items in `evidence_ids`.
    """
    spans = [
        span
        for e in case['evidence']
        if e['verified_against'] in TEXT_KINDS and (evidence_ids is None or e['id'] in evidence_ids)
        for span in [align(e, documents, source_map)]
        if span
    ]
    if not spans:
        return None
    for rank, node_id in enumerate(ranked, 1):
        v = nodes[node_id]
        if any(v['doc'] == d and v['page'] == p and v['start'] < b and a < v['end'] for d, p, a, b in spans):
            return 1 / rank
    return 0.0


def judged_at(ranked: Sequence[str], judged: Collection[str], k: int) -> float | None:
    """Share of the top k that has a relevance judgment."""
    top = ranked[:k]
    return sum(n in judged for n in top) / len(top) if top else None


def condensed(ranked: Iterable[str], judged: Collection[str]) -> list[str]:
    """The ranked list without unjudged results, for checking sensitivity to missing judgments."""
    return [n for n in ranked if n in judged]


def unmapped_rate(
    cases: Iterable[Mapping[str, Any]], documents: Documents, source_map: Mapping[str, str] | None = None
) -> float | None:
    """Share of text-layer and Git evidence whose quote is not found in the run's text."""
    items = [e for c in cases for e in c['evidence'] if e.get('verified_against') in TEXT_KINDS]
    return sum(align(e, documents, source_map) is None for e in items) / len(items) if items else None
