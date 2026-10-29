"""Shapes of the JSON values MimiRAG reads and returns, and the deployment hook interface.

Manifests, locators and evidence packets stay plain dictionaries so that they
serialize as they are; these definitions document and type-check their fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any, Literal, NamedTuple, Protocol, Required, TypedDict

SourceKind = Literal['text', 'html', 'pdf', 'git']


class Source(TypedDict, total=False):
    """One manifest entry.

    File sources have `path` and `sha256`; Git sources have `repo`, `path` and a
    commit as `version`.
    """

    id: Required[str]
    kind: Required[SourceKind]
    version: Required[str]
    path: str
    repo: str
    sha256: str
    ref: str
    pages: int | None
    ocr_pages: list[int]
    admitted_lines: list[list[int]]
    title: str
    citation: str
    family: str
    topic: str
    date: str
    qualification: str


class CodeSpec(TypedDict, total=False):
    """A code repository: one commit, an optional ref and the readable path prefixes (`scope`)."""

    repo: Required[str]
    version: Required[str]
    ref: str
    scope: Required[list[str]]


class Manifest(TypedDict, total=False):
    documents: Required[list[Source]]
    code: CodeSpec


class OcrSettings(TypedDict, total=False):
    """An external OCR command.

    `command` is an argument list in which '{images}' is replaced by a directory of
    page images; `images_at` is where the command sees that directory, for example
    inside a container.
    """

    command: Required[list[str]]
    dpi: Required[int]
    images_at: str


class Page(NamedTuple):
    """Text of a PDF page (`number` from 1) or of a whole non-PDF source (`number` None)."""

    number: int | None
    text: str


class Unit(NamedTuple):
    """A passage of whole blocks spanning 1-based lines `first` to `last` of its page."""

    first: int
    last: int
    text: str


class Extraction(NamedTuple):
    """A source's pages and, for PDFs, the sorted banner lines removed from them."""

    pages: list[Page]
    banners: list[str] | None


class PassageLocator(TypedDict):
    """Content-addressed citation of an indexed passage."""

    tool: Literal['search']
    format: int
    source: str
    version: str
    sha256: str
    page: int | None
    lines: list[int]
    extraction: str
    passage: str


class CodeLocator(TypedDict):
    """Citation of a line range in a file at one commit."""

    tool: Literal['code']
    source: str
    lines: list[int]
    sha256: str
    version: str
    repo: str
    path: str


class Packet(TypedDict):
    """What every tool returns: evidence items, how they were found and a standing qualification."""

    tool: str
    evidence: list[dict[str, Any]]
    trace: dict[str, Any]
    qualification: str


class BuildReport(TypedDict):
    index: str
    identity: str
    sources: int
    bytes: int


class Admission(Protocol):
    """Deployment hooks that add sources to a build and gate builds and queries.

    The metadata returned by `prepare` is stored in the index as given and handed
    back to `state`; MimiRAG never interprets it.
    """

    def prepare(self, manifest: Manifest, previous: Any) -> tuple[list[Source], Any]:
        """Return extra sources and metadata; `previous` is the old index's metadata or None."""
        ...

    def before_publish(self, manifest: Manifest) -> None:
        """Called just before the new index replaces the old one; raise to stop publication."""
        ...

    def state(self, manifest: Manifest, metadata: Any, indexed: Set[str]) -> tuple[Mapping[str, str | None], list[str]]:
        """Return (excluded, notices): excluded maps a source ID to its successor's ID or None."""
        ...
