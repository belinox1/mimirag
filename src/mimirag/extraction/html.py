"""HTML to plain text that keeps paragraphs and table rows."""

from __future__ import annotations

from html.parser import HTMLParser
import re

HIDDEN = frozenset({'script', 'style', 'noscript'})
OPENS_PARAGRAPH = frozenset({'p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'tr', 'br', 'section'})
CLOSES_PARAGRAPH = frozenset({'p', 'h1', 'h2', 'h3', 'h4', 'li', 'tr'})
LINE_BREAKS = frozenset({'tr', 'br'})
CELLS = frozenset({'td', 'th'})


class _TextCollector(HTMLParser):
    """Blocks become paragraphs, table rows stay on one line and hidden elements are dropped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in HIDDEN:
            self.hidden += 1
        if self.hidden:
            return
        if tag in OPENS_PARAGRAPH:
            self.parts.append('\n' if tag in LINE_BREAKS else '\n\n')
        if tag in CELLS:
            self.parts.append(' | ')

    def handle_endtag(self, tag: str) -> None:
        if tag in HIDDEN:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in CLOSES_PARAGRAPH:
            self.parts.append('\n' if tag == 'tr' else '\n\n')

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        lines = (' '.join(line.split()) for line in ''.join(self.parts).splitlines())
        return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def html_text(data: bytes) -> str:
    """Plain text of UTF-8 HTML: one space between words, blank lines between paragraphs."""
    collector = _TextCollector()
    collector.feed(data.decode('utf-8'))
    return collector.text()
