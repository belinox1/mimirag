"""Alignment of gold evidence quotes to a run's extracted text.

Quotes and text go through the same normalization, and every normalized
character maps back to raw offsets, so a match is reported as a raw span.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import re
from typing import Any
import unicodedata

from ..errors import MimiRAGError

ALIGNMENT = 'alignment-normalization-v1.2'
"""Label of the normalization rules; it changes whenever an alignment could."""

# Typographic single and double quotation marks, folded to ASCII quotes.
SINGLE_QUOTES = '\u2018\u2019\u201a\u201b'
DOUBLE_QUOTES = '\u201c\u201d\u201e\u201f'
QUOTES = str.maketrans({**dict.fromkeys(SINGLE_QUOTES, "'"), **dict.fromkeys(DOUBLE_QUOTES, '"')})
# Hyphen-minus, hyphen, non-breaking hyphen, figure dash, en dash, em dash, minus sign, soft hyphen.
DASHES = frozenset('-\u2010\u2011\u2012\u2013\u2014\u2212\u00ad')
MAX_PAGE_RANGE = 1000

Documents = Mapping[str, Mapping[Any, str]]
"""Extracted text of a run: {document key: {page number or None: text}}."""
Span = tuple[str, Any, int, int]
"""(document key, page, raw start, raw end) of an aligned quote."""


def normalize(text: str) -> tuple[str, list[int], list[int]]:
    """Normalized text with, for each of its characters, the raw start and end offsets.

    NFKC, typographic quotes folded to ASCII, a dash before a line break joins the
    words, other dashes are dropped and whitespace runs become one space; case is kept.
    """
    # NFKC is applied per base character with its combining marks, which composes Latin
    # accents as whole-string NFKC would. Hangul jamo and Indic vowel signs may differ, but
    # quotes and texts go through the same function.
    clusters = list(_clusters(text))
    cluster_end = dict(clusters)
    chars = [
        (c, start) for start, end in clusters for c in unicodedata.normalize('NFKC', text[start:end]).translate(QUOTES)
    ]
    out: list[str] = []
    starts: list[int] = []
    k = 0
    while k < len(chars):
        c, origin = chars[k]
        if c in DASHES:  # dropped, so 'e-mail' and 'email' align; ranges such as '2-3' become '23'
            k = _after_dash(chars, k)
            continue
        if not c.isspace():
            out.append(c)
            starts.append(origin)
        elif out and out[-1] != ' ':
            out.append(' ')
            starts.append(origin)
        k += 1
    if out and out[-1] == ' ':
        out.pop()
        starts.pop()
    return ''.join(out), starts, [cluster_end[s] for s in starts]


def _clusters(text: str) -> Iterator[tuple[int, int]]:
    """(start, end) of each base character together with its combining marks."""
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and unicodedata.combining(text[j]):
            j += 1
        yield i, j
        i = j


def _after_dash(chars: list[tuple[str, int]], k: int) -> int:
    """Where to continue after the dash at `k`.

    A dash ending a line (LF or CRLF) is hyphenation: skip it, the line break and the
    next line's indentation. Any other dash is skipped alone.
    """
    j = k + 1
    while j < len(chars) and chars[j][0] in ' \t\r':
        j += 1
    if j < len(chars) and chars[j][0] == '\n':
        j += 1
        while j < len(chars) and chars[j][0].isspace():
            j += 1
        return j
    return k + 1


def tokens(text: str) -> set[str]:
    """Case-folded words of the normalized text."""
    return set(re.findall(r'\w+', normalize(text)[0].casefold()))


def cited_page(evidence: Mapping[str, Any]) -> int | None:
    """First page of a locator such as 'p12' or 'pdf p3-4', or None."""
    match = re.match(r'(?:pdf\s+)?p(\d+)', evidence.get('locator', ''), re.IGNORECASE)
    return int(match.group(1)) if match else None


def cited_pages(evidence: Mapping[str, Any]) -> list[int]:
    """All pages a locator names, including ranges such as p2-4 and several p markers."""
    if cited_page(evidence) is None:
        return []
    pages: set[int] = set()
    for match in re.finditer(r'(?<!\w)p(\d+)(?:\s*[-\u2013]\s*p?(\d+))?', evidence['locator'], re.IGNORECASE):
        first = int(match[1])
        last = int(match[2] or first)
        if first < 1 or last < first or last - first > MAX_PAGE_RANGE:
            raise MimiRAGError('Invalid PDF page range')
        pages.update(range(first, last + 1))
    return sorted(pages)


def document_key(evidence: Mapping[str, Any], source_map: Mapping[str, str] | None = None) -> str:
    """Document key in the run; `source_map` renames case source IDs to index source IDs."""
    source = (source_map or {}).get(evidence['source'], evidence['source'])
    return f'{source}:{evidence["path"]}' if evidence.get('path') else source


def line_window(text: str, lines: list[int]) -> tuple[int, int]:
    """Character offsets of the 1-based inclusive line range `lines` in `text`."""
    starts = [0] + [m.end() for m in re.finditer('\n', text)]
    first, last = lines
    if not 1 <= first <= last <= len(starts):
        raise MimiRAGError(f'Line range {lines} outside the document')
    return starts[first - 1], starts[last] if last < len(starts) else len(text)


def align(
    evidence: Mapping[str, Any], documents: Documents, source_map: Mapping[str, str] | None = None
) -> Span | None:
    """Find a gold quote in the run's text and return its raw span, or None.

    The cited pages are tried first, in order. Evidence with `lines` is searched only
    within those lines, on the pages long enough to hold them. A match on another page is
    returned; scoring reports it as off-page.
    """
    key = document_key(evidence, source_map)
    if key not in documents:
        raise MimiRAGError(f'No extracted document for evidence source {key}; declare the source mapping')
    quote = normalize(evidence['quote'])[0]
    pages, cited = documents[key], set(cited_pages(evidence))
    lines, fitted = evidence.get('lines'), False
    for number in sorted(pages, key=lambda p: (p not in cited, p if p is not None else 0)):
        offset, text = 0, pages[number]
        if lines:
            try:
                offset, end = line_window(text, lines)
            except MimiRAGError:
                continue  # the page is too short for the line range
            fitted, text = True, text[offset:end]
        normalized, starts, ends = normalize(text)
        start = normalized.find(quote) if quote else -1
        if start >= 0:
            return key, number, offset + starts[start], offset + ends[start + len(quote) - 1]
    if lines and not fitted:
        raise MimiRAGError(f'Line range {lines} outside the document')
    return None
