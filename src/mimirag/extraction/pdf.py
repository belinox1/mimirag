"""PDF text from Poppler, OCR through an external command, and running-banner removal."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
import subprocess
import tempfile

from ..datatypes import OcrSettings
from ..errors import MimiRAGError

BANNER_MIN_PAGES = 3
BANNER_MIN_SHARE = 0.6
SECONDS_PER_OCR_PAGE = 120


def pdf_pages(path: Path, page: int | None = None) -> list[str]:
    """Reading-order text of every page, or of one page, from Poppler's pdftotext."""
    selection = [] if page is None else ['-f', str(page), '-l', str(page)]
    command = ['pdftotext', '-enc', 'UTF-8', *selection, str(path), '-']
    pages = _run(command, timeout=60).decode('utf-8').split('\f')
    if not pages[-1].strip():
        pages.pop()
    return pages


def ocr_pages(path: Path, numbers: Sequence[int], ocr: OcrSettings) -> dict[int, str]:
    """Text of the given PDF pages from an external OCR command.

    Each page is rendered to `p<number>.png` at `ocr['dpi']` in a temporary
    directory, together with `list.txt` naming the images in order. The command
    prints the text of each page, separated by form feeds.
    """
    with tempfile.TemporaryDirectory(prefix='ocr-') as name:
        directory = Path(name)
        for number in numbers:
            render = [
                'pdftoppm',
                '-r',
                str(ocr['dpi']),
                '-f',
                str(number),
                '-l',
                str(number),
                '-singlefile',
                '-png',
                str(path),
                str(directory / f'p{number}'),
            ]
            _run(render, timeout=SECONDS_PER_OCR_PAGE)
        seen = ocr.get('images_at', str(directory))
        listing = ''.join(f'{seen}/p{n}.png\n' for n in numbers)
        (directory / 'list.txt').write_text(listing, encoding='utf-8')
        command = [part.replace('{images}', str(directory)) for part in ocr['command']]
        timeout = SECONDS_PER_OCR_PAGE * max(1, len(numbers))
        output = _run(command, timeout=timeout)
    texts = output.decode('utf-8').split('\f')
    if len(texts) == len(numbers) + 1 and not texts[-1].strip():
        texts.pop()
    if len(texts) != len(numbers):
        raise MimiRAGError('OCR output does not have one text per page')
    return dict(zip(numbers, texts, strict=True))


def _run(command: list[str], *, timeout: int) -> bytes:
    """Standard output of an external tool; a failure or timeout raises MimiRAGError."""
    try:
        return subprocess.run(command, check=True, capture_output=True, timeout=timeout).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors='replace').strip()
        raise MimiRAGError(f'{command[0]} failed: {detail}') from error
    except subprocess.TimeoutExpired as error:
        raise MimiRAGError(f'{command[0]} timed out after {timeout} s') from error


def _margins(lines: Sequence[str]) -> set[int]:
    """Indexes of the first and last non-empty lines."""
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    return set(nonempty[:1] + nonempty[-1:])


def banner_lines(pages: Sequence[str]) -> set[str]:
    """Margin lines (first or last non-empty) repeated on at least 3 pages and 60% of all pages."""
    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.split('\n')
        counts.update({lines[i].strip() for i in _margins(lines)})
    threshold = max(BANNER_MIN_PAGES, len(pages) * BANNER_MIN_SHARE)
    return {line for line, count in counts.items() if count >= threshold}


def blank_banners(page: str, banners: Iterable[str]) -> str:
    """Empty the page's margin lines that are banners; line numbers and body text are unchanged."""
    banners = set(banners)
    lines = page.split('\n')
    for i in _margins(lines):
        if lines[i].strip() in banners:
            lines[i] = ''
    return '\n'.join(lines)
