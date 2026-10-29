"""Passage units: whole blank-line separated blocks merged up to a target size."""

from __future__ import annotations

from collections.abc import Iterator

from .datatypes import Unit

UNIT_TARGET = 1800
"""Characters a unit may reach by merging blocks; a single longer block is kept whole."""


def units(text: str, target: int = UNIT_TARGET) -> Iterator[Unit]:
    """Split `text` into units with their 1-based first and last lines.

    Blocks are merged until adding the next would pass `target` characters. A block
    is never split, so a long block becomes one oversized unit.
    """
    lines = text.splitlines()
    current: tuple[int, int] | None = None
    for start, end in _blocks(lines):
        if current and len('\n'.join(lines[current[0] : end])) > target:
            yield _unit(lines, *current)
            current = None
        current = (start, end) if current is None else (current[0], end)
    if current:
        yield _unit(lines, *current)


def _blocks(lines: list[str]) -> Iterator[tuple[int, int]]:
    """Half-open line ranges of the blocks; a blank line ends a block unless the block is empty."""
    start = 0
    for i, line in enumerate(lines):
        if not line.strip() and i > start:
            yield start, i
            start = i + 1
    if start < len(lines):
        yield start, len(lines)


def _unit(lines: list[str], start: int, end: int) -> Unit:
    return Unit(start + 1, end, '\n'.join(lines[start:end]))
