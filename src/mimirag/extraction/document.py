"""Deterministic text extraction of whole sources and of single PDF pages.

The same bytes and tools always give the same pages, so an index can store a
hash of the output and a resolver can re-extract and compare.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..datatypes import Extraction, OcrSettings, Page, Source
from ..errors import IntegrityError, MimiRAGError
from ..sources import source_bytes, source_path
from .html import html_text
from .pdf import banner_lines, blank_banners, ocr_pages, pdf_pages


def extract_document(root: Path, source: Source, ocr: OcrSettings | None = None) -> Extraction:
    """Pages of a source and, for a PDF, the banner lines removed from them.

    A non-PDF source gives one page numbered None and no banners.
    """
    data = source_bytes(root, source)  # verifies a PDF too, before external tools read it
    if source['kind'] == 'pdf':
        return _extract_pdf(root, source, ocr)
    try:
        text = html_text(data) if source['kind'] == 'html' else data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise MimiRAGError(f'Source is not UTF-8 text: {source["id"]}') from error
    if source.get('admitted_lines') is not None:
        text = _admit_lines(text, source['admitted_lines'])
    return Extraction([Page(None, text)], None)


def extract_page(
    root: Path, source: Source, number: int, banners: Sequence[str], ocr: OcrSettings | None = None
) -> str:
    """One PDF page as `extract_document` gives it, using the document's banner set.

    This relies on the reader and the OCR command giving the same text for a page
    alone as within the whole document; callers compare the result with the page
    hash stored at build time.
    """
    if type(number) is not int or number < 1:
        raise MimiRAGError('PDF page numbers start at 1')
    source_bytes(root, source)
    path = source_path(root, source)
    if number in source.get('ocr_pages', []):
        text = ocr_pages(path, [number], _require(ocr))[number]
    else:
        pages = pdf_pages(path, number)
        if len(pages) != 1:
            raise MimiRAGError('Expected the text of exactly one PDF page')
        text = pages[0]
    text = blank_banners(text, banners)
    source_bytes(root, source)
    return text


def _extract_pdf(root: Path, source: Source, ocr: OcrSettings | None) -> Extraction:
    path = source_path(root, source)
    pages = pdf_pages(path)
    if source.get('pages') is not None and len(pages) != source['pages']:
        raise IntegrityError('PDF page-count mismatch')
    numbers = source.get('ocr_pages', [])
    valid = all(type(n) is int and 1 <= n <= len(pages) for n in numbers)
    if not valid or len(set(numbers)) != len(numbers):
        raise MimiRAGError('OCR pages must be unique PDF page numbers')
    if numbers:
        recognized = ocr_pages(path, sorted(numbers), _require(ocr))
        pages = [recognized.get(i, text) for i, text in enumerate(pages, 1)]
    banners = banner_lines(pages)
    texts = [blank_banners(page, banners) for page in pages]
    # External tools read the file by path; check that it did not change meanwhile.
    source_bytes(root, source)
    return Extraction([Page(i, text) for i, text in enumerate(texts, 1)], sorted(banners))


def _admit_lines(text: str, ranges: Sequence[Sequence[int]]) -> str:
    """Empty every line outside the admitted ranges, so line numbers still match the file."""
    lines = text.splitlines()
    admitted: set[int] = set()
    for first, last in ranges:
        if not 1 <= first <= last <= len(lines):
            raise MimiRAGError('Admitted line range outside source')
        admitted.update(range(first, last + 1))
    return '\n'.join(line if i in admitted else '' for i, line in enumerate(lines, 1))


def _require(ocr: OcrSettings | None) -> OcrSettings:
    if ocr is None:
        raise MimiRAGError('Source declares OCR pages but no OCR command is configured')
    return ocr
